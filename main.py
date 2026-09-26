"""投喂亚托莉（astrbot_plugin_bot_treat）

给 Atri 发一张食物照片并说「投喂」，她会自己判断吃不吃：
  - 吃   -> 生成一张「她正在吃这份食物」的照片发回会话（复用陪伴插件生图链路，selfie 保脸）
  - 不吃 -> 只回一句符合性格的拒绝，不消耗生图配额

集成原则（勿违反）：
  - 全程只经 companion_bridge 的 getattr 防御式调用访问陪伴插件，
    不 import 其内部模块、不修改其任何文件与配置；
  - 只注册一个消息处理入口，不注册 llm_tool（避免与 reality_companion 的拍照工具撞名）；
  - 安全硬闸为纯代码判定，命中即拒绝，不交给 LLM 裁量。

设计说明（为什么只用一个 @filter.event_message_type(ALL) 入口）：
  AstrBot 的唤醒检查阶段会把唤醒前缀（默认 `/`）从 event.message_str 上剥掉，
  因此 `/投喂` 与「投喂」在插件视角收到的文本完全一致。若同时注册
  @filter.command("投喂") 与一个 ALL 入口，同一条消息会被两个 handler 各处理一次，
  造成重复生图。故此处只保留统一入口：由框架负责唤醒判定（群聊需 @ 或唤醒前缀，
  私聊默认视为已唤醒），本插件只做文本模式匹配 + 已处理/已回复去重。
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from typing import AsyncGenerator, Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register

try:  # 规范导入；失败时退回 filter 命名空间，保证插件仍可加载
    from astrbot.core.star.filter.event_message_type import EventMessageType
except Exception:  # pragma: no cover
    EventMessageType = filter.EventMessageType  # type: ignore[attr-defined]

from .companion_bridge import CompanionBridge, _clip
from .decision import (
    TreatConfig,
    decide_eat,
    forced_reason,
    get_state,
    keyword_blocked,
    load_states,
    note_feed,
    precheck_images,
    prune_states,
    recognize_food,
    save_states,
    state_path,
    today_key,
)
from .prompts import FALLBACK_TEXTS, build_eat_prompt

PLUGIN_NAME = "astrbot_plugin_bot_treat"
VERSION = "0.4.0"

# 「本轮出站结果是本插件自己的」标记。投喂窗口内靠它区分「我们的进食图」与
# 「陪伴插件对照片的迟到点评/表情包」——只放行自己的，其余丢弃。
ATR_OWN_EXTRA = "bot_treat_own"
# 我们自己的结果发出后，窗口再延长多久收口（覆盖她在我们之后才吐出来的回复）
FEED_WINDOW_TAIL_SEC = 20.0

def _component_names(event: AstrMessageEvent, limit: int = 8) -> str:
    """诊断用：列出本轮消息链的组件类名，便于排查"图片没被识别到"。"""
    try:
        chain = getattr(getattr(event, "message_obj", None), "message", None) or []
    except Exception:
        return "?"
    names: list[str] = []
    for comp in chain:
        if isinstance(comp, dict):
            names.append(f"dict:{str(comp.get('type') or '?')}")
        else:
            names.append(comp.__class__.__name__)
    return ",".join(names[:limit]) or "-"


def _session_key(event: AstrMessageEvent) -> str:
    """会话键：投喂事件与照片事件必须能对上，故统一用 unified_msg_origin。"""
    return str(getattr(event, "unified_msg_origin", "") or "")


def _set_own_flag(event: AstrMessageEvent, value: bool = True) -> None:
    setter = getattr(event, "set_extra", None)
    if callable(setter):
        try:
            setter(ATR_OWN_EXTRA, value)
            return
        except Exception:
            pass
    try:
        setattr(event, ATR_OWN_EXTRA, value)
    except Exception:
        pass


def _own_flag(event: AstrMessageEvent) -> bool:
    getter = getattr(event, "get_extra", None)
    if callable(getter):
        try:
            if getter(ATR_OWN_EXTRA):
                return True
        except Exception:
            pass
    return bool(getattr(event, ATR_OWN_EXTRA, False))


def _is_private_event(event: AstrMessageEvent) -> bool:
    try:
        return not event.get_group_id()
    except Exception:
        return False


def _is_plain_photo_event(event: AstrMessageEvent, text: str) -> bool:
    """是不是「私聊 + 带图 + 没有投喂字样」的照片消息（延迟裁决的适用对象）。"""
    if not _has_image_hint(event):
        return False
    if START_RE.match(text.lstrip("/!！").strip()):
        return False
    try:
        if event.get_group_id():
            return False
    except Exception:
        return False
    return True


# 触发词：句首匹配，且必须在「本轮带图」的前提下才生效（见 _handle）。
# 刻意排除两类歧义词：
#   - 裸「喂你」——「喂你看看这个」这类口头语不该被接管；
#   - 「喂你吃 / 喂我吃」——「喂你吃饭了吗」会被误匹配（吃饭的"吃"）。
# 真正的保护是"带图才介入"，触发词只需覆盖明确表达。
START_RE = re.compile(r"^\s*(投喂|喂食|喂饭|给你吃|请你吃|尝尝这个|吃这个)")
DEDUP_TTL_SEC = 90.0
STATUS_OK = {"ok", "success", "succeeded"}


def _raw_text(event: AstrMessageEvent) -> str:
    """尽力取平台原始文本（含唤醒前缀）；取不到返回空串。

    用于区分「斜杠命令通道」与「自然语言通道」——框架剥离唤醒前缀后
    两者文本相同，只能从平台原始事件里还原。
    """
    obj = getattr(event, "message_obj", None)
    raw = getattr(obj, "raw_message", None)
    if isinstance(raw, dict):
        for key in ("raw_message", "message"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _is_slash(text: str) -> bool:
    return text.lstrip().startswith(("/", "!", "！"))


def _has_image_hint(event: AstrMessageEvent) -> bool:
    """低成本判断本轮消息是否带图（直发图 / 引用消息里的图 / 陪伴插件已解析的图源）。

    这一步必须在「接管事件」之前做：否则「喂你吃饭了吗」这类普通聊天会被
    误判成投喂、被抢走回复。
    """
    try:
        chain = getattr(getattr(event, "message_obj", None), "message", None) or []
    except Exception:
        chain = []

    def _name(obj) -> str:
        if isinstance(obj, dict):
            return str(obj.get("type") or "").lower()
        return obj.__class__.__name__.lower()

    for comp in chain:
        if _name(comp) == "image":
            return True
    for comp in chain:
        if _name(comp) == "reply":
            for sub in (getattr(comp, "chain", None) or []):
                if _name(sub) == "image":
                    return True
    if getattr(event, "private_companion_delayed_image_sources", None):
        return True
    return False


@register(
    PLUGIN_NAME,
    "Meru",
    "投喂亚托莉：发食物照片给她，按性格与状态自主判断吃不吃，吃则生成进食照片",
    VERSION,
    "https://github.com/Meru-tea/astrbot_plugin_bot_treat",
)
class BotTreatPlugin(Star):
    """投喂亚托莉：发一张食物照片给我并说「投喂」，我会自己决定吃不吃。

    吃就把我吃这份食物的样子发给你；不想吃的话，我会告诉你为什么～

    用法：
      投喂 + 食物照片            （私聊建议用 /投喂 开头）
      触发词：投喂 / 喂食 / 喂饭 / 喂你吃 / 喂我吃 / 给你吃 / 请你吃 / 尝尝这个 / 吃这个
      必须带图——不带图的消息我完全不会介入，你还是可以照常跟我聊天。
    """

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._data_dir = self._resolve_data_dir()
        try:
            os.makedirs(self._data_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"bot_treat: 数据目录创建失败: {e}")
        self.bridge = CompanionBridge(context)
        self._seen: dict[str, float] = {}
        # 会话键 -> 抑制窗口到期时间戳：窗口内只放行本插件自己的出站结果
        self._feed_windows: dict[str, float] = {}
        # 会话键 -> 事件：照片消息的「延迟裁决」用，被紧随其后的「投喂」唤醒
        self._photo_holds: dict[str, asyncio.Event] = {}
        # 会话键 -> (paths, ts)：照片挂起时预先落好的图，供随后那条「投喂」取用
        self._held_photos: dict[str, tuple[list[str], float]] = {}
        # 闸门被调用的计数（自证用：只记前 30 次，区分"没被调用"与"调用了但早退"）
        self._gate_calls: int = 0
        self._states: dict = load_states(state_path(self._data_dir))

    # -------------------------------------------------- 生命周期

    async def initialize(self):
        cfg = self.cfg()
        logger.info(
            f"bot_treat: 投喂亚托莉 v{VERSION} 已加载"
            f"（数据目录 {self._data_dir}，陪伴插件桥"
            f"{'可用' if self.bridge.available() else '暂不可用（稍后自动重试）'}）"
        )
        # 参考图策略：use_food_as_ref 时同时传「人设图 + 食物图」并用序数语法分派角色
        # （第1张 identity / 第2张 scene），人设图取不到时自动退回不传参考图。
        logger.info(
            f"bot_treat: 参考图策略 use_food_as_ref={cfg.use_food_as_ref} "
            f"gen_kind={cfg.gen_kind}"
            + ("（食物图将作为第2张 scene 参考参与生成）" if cfg.use_food_as_ref else "（仅用陪伴插件自动人设图）")
        )
        logger.info(
            f"bot_treat: 唯一回复策略 照片挂起={cfg.photo_hold_sec}s "
            f"投喂后等图={cfg.feed_wait_photo_sec}s 抑制窗口={cfg.feed_watch_sec}s "
            f"回看={cfg.image_lookback_sec}s"
        )
        self._log_gate_registration()

    def _log_gate_registration(self) -> None:
        """启动自检：确认「出站闸门」真的注册进了寄存器。

        为什么要这个：整条唯一回复方案都压在 on_decorating_result 上，而它若没注册，
        运行时**完全没有任何迹象**（只在丢弃时才打日志）——实测就踩过这个坑，
        白跑了一轮用户测试。这里开局就报一句话，把"没注册"变成一眼可见。
        """
        try:
            from astrbot.core.star.star_handler import (  # type: ignore
                EventType,
                star_handlers_registry,
            )
        except Exception as e:
            logger.warning(f"bot_treat: 无法自检钩子注册（import 失败）: {_clip(e, 120)}")
            return
        for label, ev_type in (
            ("出站闸门", getattr(EventType, "OnDecoratingResultEvent", None)),
        ):
            if ev_type is None:
                logger.warning(f"bot_treat: {label} 自检失败：拿不到事件类型枚举")
                continue
            try:
                all_h = star_handlers_registry.get_handlers_by_event_type(ev_type)
                mine = [
                    h
                    for h in all_h
                    if "bot_treat" in str(getattr(h, "handler_module_path", ""))
                ]
                logger.info(
                    f"bot_treat: {label}注册自检 → 本插件={len(mine)} 同类总数={len(all_h)} "
                    f"名称={[getattr(h, 'handler_name', '?') for h in mine]} "
                    f"优先级={[getattr(h, 'extras_configs', {}).get('priority') for h in mine]}"
                )
                if not mine:
                    logger.warning(
                        f"bot_treat: ⚠️ {label}未注册成功——唯一回复保障将不生效，请立即排查"
                    )
            except Exception as e:
                logger.warning(f"bot_treat: {label} 自检异常: {_clip(e, 160)}")

    async def terminate(self):
        save_states(state_path(self._data_dir), self._states)
        self._feed_windows.clear()
        self._held_photos.clear()
        for evt in list(self._photo_holds.values()):
            try:
                evt.set()
            except Exception:
                pass
        self._photo_holds.clear()
        logger.info("bot_treat: 投喂亚托莉已卸载")

    # -------------------------------------------------- 出站结果闸门（发送前最后一道）

    @filter.on_decorating_result(priority=1000)
    async def outbound_gate(self, event: AstrMessageEvent, *args, **kwargs):
        """投喂回合的「唯一回复」保障。

        背景（2026-09-25 实测）：陪伴插件对"只带图"的私聊消息会**立刻发一条反应表情包**，
        并在 8 秒防抖到期后把「延迟图片」注入被动状态管线、走它自己的常规 LLM 回复。
        **这条链路与原始事件有没有被 stop_event 无关**（实测我们 21:33:45 就 stop 了那条消息，
        它 21:34:54 照样生成点评）。所以只能在**发送前**拦。

        优先级 1000：排在陪伴插件会「挂图/发送」的钩子（0 / -10000 / -18000 / -20000）之前，
        也排在她的 20000/10000（群安全标记、主动消息桥）之后——本插件不受那两条影响。
        """
        cfg = self.cfg()
        if not cfg.enable_feed:
            return
        key = _session_key(event)
        if not key:
            return
        now = time.time()
        own = _own_flag(event)
        until = self._feed_windows.get(key, 0.0)

        # 入口自证：只记前若干次，用来区分「闸门没被调用」与「调用了但提前返回」
        self._gate_calls += 1
        if self._gate_calls <= 30:
            logger.info(
                f"bot_treat: 闸门被调用 #{self._gate_calls} key={key} "
                f"私聊={_is_private_event(event)} own={own} 含图={_has_image_hint(event)}"
            )

        # 诊断：私聊出站结果每次记一行（低频），用于确认闸门到底有没有被调用
        if _is_private_event(event):
            logger.info(
                f"bot_treat: 闸门 key={key} own={own} 窗口剩余={max(0.0, until - now):.0f}s "
                f"含图={_has_image_hint(event)} "
                f"照片事件={_is_plain_photo_event(event, str(getattr(event, 'message_str', '') or ''))} "
                f"文本={_clip(getattr(event, 'message_str', ''), 20)!r} "
                f"组件={_component_names(event)}"
            )

        # 1) 我们自己的结果：放行；并把窗口收窄到「交付后一小段」
        if own:
            if until:
                self._feed_windows[key] = min(until, now + FEED_WINDOW_TAIL_SEC)
            return

        # 2) 投喂窗口内：丢弃她的迟到点评 / 反应表情包
        if until > now:
            logger.info(
                f"bot_treat: 抑制投喂窗口内的出站消息 key={key} 组件={_component_names(event)}"
            )
            try:
                event.clear_result()
                event.stop_event()
            except Exception:
                pass
            return

        # 3) 照片消息的「延迟裁决」：等一小会儿，看是否紧跟一条「投喂」
        if cfg.photo_hold_sec <= 0:
            return
        text = str(getattr(event, "message_str", "") or "")
        if not _is_plain_photo_event(event, text):
            return

        holder = self._photo_holds.get(key)
        if holder is None or holder.is_set():
            holder = asyncio.Event()
            self._photo_holds[key] = holder
        try:
            signalled = await asyncio.wait_for(
                holder.wait(), timeout=float(cfg.photo_hold_sec)
            )
        except asyncio.TimeoutError:
            signalled = False
        except Exception:
            signalled = False
        finally:
            if self._photo_holds.get(key) is holder:
                self._photo_holds.pop(key, None)
        if signalled:
            logger.info(f"bot_treat: 照片后紧跟投喂，丢弃照片那条的即时回复 key={key}")
            try:
                event.clear_result()
                event.stop_event()
            except Exception:
                pass

    # -------------------------------------------------- 小工具

    def _resolve_data_dir(self) -> str:
        try:
            return str(StarTools.get_data_dir(PLUGIN_NAME))
        except Exception:
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            return os.path.join(root, "plugin_data", PLUGIN_NAME)

    def cfg(self) -> TreatConfig:
        return TreatConfig.from_raw(self.config)

    def _gc_seen(self, now: float) -> None:
        if len(self._seen) < 200:
            return
        for key, ts in list(self._seen.items()):
            if now - ts > DEDUP_TTL_SEC:
                self._seen.pop(key, None)

    def _persist(self) -> None:
        save_states(state_path(self._data_dir), self._states)

    async def _hold_photo_event(self, event: AstrMessageEvent, cfg: TreatConfig) -> None:
        """照片消息上的「延迟裁决」（handler 层，唯一回复的主机制）。

        为什么放在 handler 层：实测我方 handler 的执行时刻（12.046）**早于**她创建
        8 秒防抖缓冲（12.056），而 `star_request.py` 的 handler 循环每步都检查
        `is_stopped()` —— 只要我们在等待后 `stop_event()`，**她的 handler 根本不会执行**：
        不建防抖缓冲、不发反应表情包、不生成照片点评。这比"事后在发送前拦截"干净得多。

        若超时未等到「投喂」：**不 stop、不干预**，她的正常流程照走（只是那条即时回复
        晚 photo_hold_sec 秒）。
        """
        if cfg.photo_hold_sec <= 0:
            return
        key = _session_key(event)
        if not key:
            return
        # 先把图落好盘：一旦我们 stop 了这条事件，她的 handler 不会再落盘，
        # 随后那条「投喂」就必须靠这里备好的路径取图。
        user_id = self.bridge.canonical_user_id(event)
        try:
            paths = await self.bridge.persist_images(event, user_id)
        except Exception as e:
            logger.debug(f"bot_treat: 照片挂起前落盘失败: {_clip(e, 160)}")
            paths = []
        holder = asyncio.Event()
        self._photo_holds[key] = holder
        if paths:
            self._held_photos[key] = (list(paths), time.time())
        try:
            signalled = await asyncio.wait_for(
                holder.wait(), timeout=float(cfg.photo_hold_sec)
            )
        except asyncio.TimeoutError:
            signalled = False
        except Exception:
            signalled = False
        finally:
            if self._photo_holds.get(key) is holder:
                self._photo_holds.pop(key, None)
        if signalled:
            logger.info(
                f"bot_treat: 照片后紧跟投喂 → handler 层压掉照片那条"
                f"（她的防抖缓冲/即时回复都不会产生）key={key}"
            )
            try:
                event.stop_event()
                event.should_call_llm(True)
            except Exception:
                pass
        else:
            logger.debug(f"bot_treat: 照片后无投喂，放行她的正常回复 key={key}")
            self._held_photos.pop(key, None)

    def _open_window(self, key: str, cfg: TreatConfig) -> None:
        if key and cfg.feed_watch_sec > 0:
            self._feed_windows[key] = time.time() + float(cfg.feed_watch_sec)

    def _close_window(self, key: str) -> None:
        if key:
            self._feed_windows.pop(key, None)

    def _arm_feed(self, event: AstrMessageEvent, cfg: TreatConfig) -> None:
        """进入投喂回合：唤醒在等的照片事件、打开抑制窗口、把自己标记为「我方事件」。

        必须在**接管事件的同一时刻**调用：窗口开得越早，她那条约 1.4 秒后到达的即时
        回复才越可能被拦下；而 own 标记是窗口内放行「我们自己的进食图」的唯一凭据。
        """
        key = _session_key(event)
        self._open_window(key, cfg)
        holder = self._photo_holds.get(key)
        if holder is not None and not holder.is_set():
            holder.set()
        _set_own_flag(event, True)

    async def _await_photo(
        self,
        user_id: str,
        raw_sender: str,
        cfg: TreatConfig,
    ) -> list[str]:
        """「先说投喂、再发照片」的等待：轮询回看窗口，看照片有没有随后到达。

        轮询而不是靠事件唤醒，是因为照片那条事件不一定产生出站结果（没结果就不会触发
        出站闸门），靠它来唤醒不可靠；而照片一落盘就一定能被回看扫到。
        """
        if cfg.feed_wait_photo_sec <= 0:
            return []
        keys = [user_id, raw_sender]
        deadline = time.time() + float(cfg.feed_wait_photo_sec)
        while time.time() < deadline:
            await asyncio.sleep(0.5)
            try:
                got = await self.bridge.recent_persisted_images(
                    keys, float(cfg.image_lookback_sec)
                )
            except Exception:
                got = []
            if got:
                logger.info(f"bot_treat: 等到了照片 {len(got)} 张（先投喂、后发图）")
                return got
        return []

    # -------------------------------------------------- 统一入口

    @filter.event_message_type(EventMessageType.ALL)
    async def feed_entry(self, event: AstrMessageEvent) -> AsyncGenerator:
        try:
            async for result in self._handle(event):
                yield result
        except Exception as e:  # 任何未预期异常都不该让消息静默丢失
            logger.warning(f"bot_treat: 处理投喂异常: {_clip(e, 300)}")
            try:
                yield event.plain_result(FALLBACK_TEXTS["bridge_down"])
            except Exception:
                pass

    async def _handle(self, event: AstrMessageEvent) -> AsyncGenerator:
        cfg = self.cfg()
        if not cfg.enable_feed:
            return

        text = str(getattr(event, "message_str", "") or "").strip()
        # 去掉可能残留的命令前缀（部分适配器不剥离）
        text = text.lstrip("/!！").strip()
        if not START_RE.match(text):
            if _is_private_event(event) and _has_image_hint(event):
                logger.info(
                    f"bot_treat: 看到私聊照片消息（无触发词）组件={_component_names(event)}"
                )
                await self._hold_photo_event(event, cfg)
            return

        # 已被其他 handler 处理/回复过的，直接让行，避免重复生图
        if getattr(event, "_has_send_oper", False):
            return
        try:
            if event.is_stopped():
                return
        except Exception:
            pass
        try:
            if event.get_result() is not None:
                return
        except Exception:
            pass

        # 场景开关
        try:
            group_id = event.get_group_id()
        except Exception:
            group_id = ""
        if group_id and not cfg.enable_group:
            return
        if not group_id and not cfg.enable_private:
            return

        # 通道与「是否值得接管」判定
        raw = _raw_text(event)
        slash = _is_slash(raw) if raw else True
        hint = _has_image_hint(event)

        user_id = self.bridge.canonical_user_id(event)
        try:
            raw_sender = str(event.get_sender_id() or "")
        except Exception:
            raw_sender = ""

        # 回看最近收到的图：覆盖两种情况
        #   a) 照片与「投喂」分两条消息发（陪伴插件已把图落盘，与本插件是否被触发无关）
        #   b) 引用消息里的图无法本地化时，还能用最近那张兜底
        # 成本只有一次 os.listdir，故每次触发都扫。
        lookback: list[str] = []
        if cfg.image_lookback_sec > 0:
            try:
                lookback = await self.bridge.recent_persisted_images(
                    [user_id, raw_sender],
                    float(cfg.image_lookback_sec),
                )
            except Exception as e:
                logger.debug(f"bot_treat: 回看最近图片失败: {_clip(e, 160)}")
                lookback = []

        # 触发命中后打一条诊断日志（低频，用于排查"为什么不触发/为什么说没图"）
        logger.info(
            f"bot_treat: 触发命中 user={user_id or raw_sender} text={_clip(text, 40)!r} "
            f"slash={slash} 消息含图迹象={hint} 回看={len(lookback)} "
            f"组件={_component_names(event)}"
        )

        if not hint and not lookback:
            # 没有图：先别急着回「要带图」——可能是「先说投喂、再发照片」。
            # 先开窗口等她随后要发的照片（那期间她若对照片发即时回复会被闸门拦下）；
            # 若最终没等到，再按渠道决定是否接管；自然语言渠道完全不介入（不抢闲聊）。
            if not slash and not cfg.enable_natural_language:
                return
            key = _session_key(event)
            self._open_window(key, cfg)
            lookback = await self._await_photo(user_id, raw_sender, cfg)
            if not lookback:
                if not slash:
                    self._close_window(key)
                    return
                self._arm_feed(event, cfg)
                try:
                    event.stop_event()
                    event.should_call_llm(True)
                except Exception:
                    pass
                yield event.plain_result(FALLBACK_TEXTS["no_image"])
                return
            # 等到了照片：继续按「有图」走下面的流程

        # 有图但明确关掉了自然语言通道时，只有斜杠命令可用
        if not slash and not cfg.enable_natural_language:
            return

        # 消息去重
        now = time.time()
        self._gc_seen(now)
        message_id = str(getattr(getattr(event, "message_obj", None), "message_id", "") or "")
        if message_id:
            if message_id in self._seen:
                return
            self._seen[message_id] = now

        # 抢占：本插件接管后，必须同时掐掉「事件继续传播」与「框架默认 LLM 回复」两条路。
        # 只 stop_event 不够——后续 yield 新结果可能覆盖 stopped 状态，导致框架再让主模型回一遍。
        self._arm_feed(event, cfg)
        try:
            event.stop_event()
        except Exception:
            pass
        try:
            event.should_call_llm(True)
        except Exception:
            pass

        async for result in self._run_feed(event, text, cfg, user_id, raw_sender, lookback):
            yield result

    # -------------------------------------------------- 主流程

    async def _run_feed(
        self,
        event,
        text: str,
        cfg: TreatConfig,
        user_id: str = "",
        raw_sender: str = "",
        lookback: Optional[list[str]] = None,
    ) -> AsyncGenerator:
        if not self.bridge.available():
            yield event.plain_result(FALLBACK_TEXTS["bridge_down"])
            return

        user_id = user_id or self.bridge.canonical_user_id(event)
        umo = str(getattr(event, "unified_msg_origin", "") or "")

        # 1) 取图：优先用「照片挂起」时预先落好的图（那条照片事件已被我们 stop，
        #    她的 handler 不会再落盘，回看目录里也就没有它）；其次本轮消息（**含引用消息里的图**，
        #    七级阶梯兜底）；最后「回看最近收到的图」（照片与文字分两条消息发的情况）。
        paths: list[str] = []
        from_hold = False
        held = self._held_photos.pop(_session_key(event), None)
        if held and (time.time() - held[1]) <= 120.0:
            paths = [p for p in held[0] if os.path.isfile(p)]
            from_hold = bool(paths)
        if not paths:
            try:
                paths = await asyncio.wait_for(
                    self.bridge.persist_images(event, user_id),
                    timeout=float(cfg.vision_timeout_sec) + 15.0,
                )
            except asyncio.TimeoutError:
                logger.warning(f"bot_treat: 取图超时 user={user_id}")
                paths = []
            if paths:
                logger.info(f"bot_treat: 取图 {len(paths)} 张（本轮消息/引用消息）")
        if not paths:
            paths = [p for p in (lookback or []) if os.path.isfile(p)]
            if paths:
                logger.info(f"bot_treat: 取图 {len(paths)} 张（回看最近收到的图）")
        if from_hold:
            logger.info(f"bot_treat: 取图 {len(paths)} 张（照片挂起时预先落盘）")
        if not paths:
            logger.info(f"bot_treat: 无可用图片 user={user_id}")
            yield event.plain_result(FALLBACK_TEXTS["no_image"])
            return

        # 2) 安全硬闸（纯代码）
        blocked = precheck_images(paths, cfg.min_image_bytes)
        if blocked:
            logger.info(f"bot_treat: 硬闸拦截({blocked}) user={user_id}")
            yield event.plain_result(FALLBACK_TEXTS.get(blocked, FALLBACK_TEXTS["non_food"]))
            return
        if keyword_blocked(text, cfg.extra_block_keywords):
            logger.info(f"bot_treat: 关键词硬闸拦截 user={user_id}")
            yield event.plain_result(FALLBACK_TEXTS["blocked_keyword"])
            return

        # 3) 食物识别
        food = await recognize_food(self.bridge, paths, cfg)
        if food is None:
            yield event.plain_result(FALLBACK_TEXTS["unreadable"])
            return
        if food.confidence < 0.3 and food.name == "看不清":
            yield event.plain_result(FALLBACK_TEXTS["unreadable"])
            return
        if not food.edible:
            today = today_key()
            prune_states(self._states, today)
            note_feed(self._states, user_id, today, accepted=False)
            self._persist()
            logger.info(f"bot_treat: 非食物/危险物({food.danger_text()}) user={user_id}")
            yield event.plain_result(self._non_food_text(food))
            return

        # 4) 状态与情境
        today = today_key()
        prune_states(self._states, today)
        state = get_state(self._states, user_id, today)
        forced = forced_reason(state, cfg)

        # 5) 吃不吃决策
        decision = await decide_eat(
            self.bridge,
            food,
            state,
            cfg,
            self.bridge.persona_excerpt(),
            forced,
        )
        logger.info(
            f"bot_treat: 决策 user={user_id} food={food.name} conf={food.confidence:.2f} "
            f"decision={decision.decision} forced={forced or '-'} llm_failed={decision.llm_failed} "
            f"reason={_clip(decision.reason, 40)}"
        )

        # 6) 拒绝分支：只回文本，不生图
        if decision.decision != "eat":
            note_feed(self._states, user_id, today, accepted=False)
            self._persist()
            if cfg.enable_memory_writeback:
                await self.bridge.memory_writeback(
                    f"主人投喂{food.name}，我没吃：{decision.reason or '不想吃'}",
                    tags=["feed", "persona_life", "food", food.name, "refused"],
                    session_id=umo,
                )
            yield event.plain_result(decision.reply_text)
            return

        # 7) 接受分支：生图
        #
        # 关于参考图（关键，改动前先读 README §4.1 与 prompts.REFERENCE_ROLE_SUFFIX）：
        #   陪伴插件在「本轮带了显式参考图」时不会再自动追加 Atri 人设图
        #   （proactive_message.py 约 16263 行的分支要求 not candidates and not paths）。
        #   所以要让食物照片参与生成，就必须把**人设图也一并传**，并用「第1张/第2张」
        #   序数角色说明逐张指定角色 —— 否则食物图会被默认标成 identity，把脸挤掉。
        reference_paths: list[str] = []
        with_food_ref = False
        if cfg.use_food_as_ref:
            persona_path = await self.bridge.persona_reference_path(
                cfg.persona_reference_image_path
            )
            if persona_path:
                reference_paths = [persona_path, *paths][:2]
                with_food_ref = len(reference_paths) >= 2
                logger.info(
                    f"bot_treat: 参考图 第1张=人设({_clip(persona_path, 80)}) "
                    f"第2张=食物({_clip(paths[0], 80)})"
                )
            else:
                logger.warning(
                    "bot_treat: use_food_as_ref=true 但未取到 Atri 人设参考图，"
                    "本次退回「不传参考图」以保住脸一致（陪伴插件会自动上人设图）"
                )

        prompt = build_eat_prompt(
            food.name,
            food.appearance,
            decision.eat_scene_prompt,
            with_food_reference=with_food_ref,
        )
        receipt = await self.bridge.generate_photo(
            event,
            prompt=prompt,
            kind=cfg.gen_kind,
            reference_image_paths=reference_paths or None,
            caption=decision.reply_text,
            send=not cfg.dry_run,
            timeout=float(cfg.photo_timeout_sec),
            api_base_url=cfg.photo_api_base_url,
            api_key=cfg.photo_api_key,
            api_model=cfg.photo_api_model,
            api_size=cfg.photo_api_size,
            api_override_sec=float(cfg.photo_api_override_sec),
        )
        status = str(receipt.get("status") or "").lower()
        generated = bool(receipt.get("generated")) or status in STATUS_OK
        # reference_roles 是「有没有用上 Atri 身份参考图」的运行时证据，务必记日志
        roles = receipt.get("reference_roles")
        logger.info(
            f"bot_treat: 生图回执 status={status} generated={generated} "
            f"reference_roles={roles} food={food.name} kind={cfg.gen_kind} "
            f"use_food_as_ref={cfg.use_food_as_ref}"
        )

        note_feed(self._states, user_id, today, accepted=generated)
        self._persist()

        if cfg.dry_run:
            logger.info(
                f"bot_treat: [dry-run] food={food.name} status={status} "
                f"roles={roles} receipt={_clip(receipt, 400)}"
            )
            yield event.plain_result(
                f"[dry-run] {food.name}｜decision={decision.decision}｜status={status or '?'}"
                f"｜roles={roles or '?'}｜scene={_clip(decision.eat_scene_prompt, 60)}"
            )
            return

        if generated:
            # 图片已由生图工具连同 caption 发出，是本次唯一的可见回复
            # （默认 LLM 链路已在入口处掐掉，此处无需重复处理）
            if cfg.enable_memory_writeback:
                await self.bridge.memory_writeback(
                    f"主人投喂{food.name}，我吃了。{decision.emotion}",
                    tags=["feed", "persona_life", "food", food.name, "accepted"],
                    session_id=umo,
                )
            return

        # 生图失败：只说「没吃成」，**绝不复用 decision.reply_text**
        #（那句话是以"已经吃了"为前提写的，配上失败会变成假的完成暗示）
        if status in ("quota_exhausted", "unauthorized"):
            text_out = FALLBACK_TEXTS["quota_exhausted"]
        else:
            text_out = FALLBACK_TEXTS["gen_failed"]
        logger.info(f"bot_treat: 生图未成功 status={status} detail={_clip(receipt.get('message'), 200)}")
        yield event.plain_result(text_out)

    # -------------------------------------------------- 文案

    @staticmethod
    def _non_food_text(food) -> str:
        danger = food.danger_text()
        if "非食物" in danger or food.category == "非食物":
            return "喂喂，这又不是食物，我可是高性能的，别想糊弄我！"
        if "酒精" in danger:
            return "这个不行啦，我可是还没到能喝酒的年纪呢，哼。"
        if "药物" in danger:
            return "药才不能乱吃呢，你想让我的核心出故障吗？"
        if "活体" in danger:
            return "活的怎么能吃呀，喂喂，先把它放回去！"
        return f"这个不行啦（{danger}），换一个给我嘛。"
