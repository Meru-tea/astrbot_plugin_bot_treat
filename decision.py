"""投喂判定与状态（astrbot_plugin_bot_treat）。

三层判定（从硬到软）：
  1. 安全硬闸 —— 纯代码，命中即拒绝，不进 LLM、不生图；
  2. 状态与情境 —— 冷却 / 饱腹 / 每日上限，由代码算成事实喂给 LLM，并强制拒绝；
  3. 人设即兴 —— LLM 在给定事实上决定吃不吃、怎么说。
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from astrbot.api import logger

from .companion_bridge import _flatten_json
from .prompts import (
    FALLBACK_TEXTS,
    FOOD_RECOGNITION_SYSTEM,
    FOOD_RECOGNITION_USER,
    build_decision_system,
    build_decision_user,
)

STATE_FILE_NAME = "feed_state.json"
STATE_KEEP_DAYS = 7


# ---------------------------------------------------------------- 配置

def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on", "是", "开启"):
        return True
    if text in ("0", "false", "no", "n", "off", "否", "关闭"):
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def _as_text(value: Any, default: str = "") -> str:
    """配置里的字符串：去空白，None/异常回退默认。"""
    if value is None:
        return default
    try:
        return str(value).strip()
    except Exception:
        return default


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,，\s]+", value) if part.strip()]
    return []


@dataclass
class TreatConfig:
    enable_feed: bool = True
    enable_private: bool = True
    enable_group: bool = True
    enable_natural_language: bool = True
    daily_limit: int = 5
    cooldown_sec: int = 30
    satiety_threshold: int = 3
    gen_kind: str = "selfie"
    # 默认开：同时传两张参考图并用序数语法逐张指定角色
    # （第1张=角色人设身份图保脸，第2张=用户食物照片仅作 scene 参考）。
    # 详见 prompts.REFERENCE_ROLE_SUFFIX 与 README §4.1。
    use_food_as_ref: bool = True
    # 人物身份参考图（本地绝对路径或 URL）；留空 = 自动用陪伴插件配置的人物参考图
    persona_reference_image_path: str = ""
    # 留空 = 用 AstrBot 默认 provider / 陪伴插件配置的视觉模型（发布版默认留空）
    vision_provider_id: str = ""
    vision_timeout_sec: int = 30
    llm_timeout_sec: int = 45
    # 生图单独的超时：带参考图的在线图片 API 实测可达 1-3 分钟，
    # 绝不能复用 llm_timeout_sec（45s 会把生图掐断在提交后）。
    photo_timeout_sec: int = 240
    # 仅本插件投喂时使用的生图接口（全空 = 完全沿用陪伴插件现有配置）
    photo_api_base_url: str = ""
    photo_api_key: str = ""
    photo_api_model: str = ""
    photo_api_size: str = ""
    # 接口覆盖的保持秒数（地址解析在生图开始后 ~20ms 完成）
    photo_api_override_sec: int = 15
    min_image_bytes: int = 6144
    # 照片与「投喂」分两条消息发时，回看最近 N 秒内该用户收到的图（0=关闭）
    image_lookback_sec: int = 120
    # 「先说投喂、再发照片」：投喂到达后最多等这么多秒照片（0=关闭）
    feed_wait_photo_sec: int = 8
    # 「先发照片、再说投喂」：照片那条的即时回复最多挂起这么多秒等投喂（0=关闭）
    photo_hold_sec: int = 6
    # 投喂回合的出站抑制窗口上限；窗口内只放行本插件自己的进食图（0=关闭）
    feed_watch_sec: int = 150
    extra_block_keywords: list = field(default_factory=list)
    enable_memory_writeback: bool = True
    dry_run: bool = False

    @classmethod
    def from_raw(cls, raw: Any) -> "TreatConfig":
        get = getattr(raw, "get", None)

        def _get(key: str, default: Any) -> Any:
            if callable(get):
                try:
                    value = raw.get(key)
                except Exception:
                    value = None
                return default if value is None else value
            return default

        kind = str(_get("gen_kind", "selfie") or "selfie").strip().lower()
        if kind not in ("selfie", "edit", "text2img"):
            kind = "selfie"
        return cls(
            enable_feed=_as_bool(_get("enable_feed", True), True),
            enable_private=_as_bool(_get("enable_private", True), True),
            enable_group=_as_bool(_get("enable_group", True), True),
            enable_natural_language=_as_bool(_get("enable_natural_language", True), True),
            daily_limit=max(1, _as_int(_get("daily_limit", 5), 5)),
            cooldown_sec=max(0, _as_int(_get("cooldown_sec", 30), 30)),
            satiety_threshold=max(1, _as_int(_get("satiety_threshold", 3), 3)),
            gen_kind=kind,
            use_food_as_ref=_as_bool(_get("use_food_as_ref", True), True),
            persona_reference_image_path=_as_text(_get("persona_reference_image_path", "")),
            vision_provider_id=str(_get("vision_provider_id", "") or "").strip(),
            vision_timeout_sec=max(5, _as_int(_get("vision_timeout_sec", 30), 30)),
            llm_timeout_sec=max(10, _as_int(_get("llm_timeout_sec", 45), 45)),
            photo_timeout_sec=max(30, _as_int(_get("photo_timeout_sec", 240), 240)),
            photo_api_base_url=_as_text(_get("photo_api_base_url", "")),
            photo_api_key=_as_text(_get("photo_api_key", "")),
            photo_api_model=_as_text(_get("photo_api_model", "")),
            photo_api_size=_as_text(_get("photo_api_size", "")),
            photo_api_override_sec=max(1, _as_int(_get("photo_api_override_sec", 15), 15)),
            min_image_bytes=max(0, _as_int(_get("min_image_bytes", 6144), 6144)),
            image_lookback_sec=max(0, _as_int(_get("image_lookback_sec", 120), 120)),
            feed_wait_photo_sec=max(0, _as_int(_get("feed_wait_photo_sec", 8), 8)),
            photo_hold_sec=max(0, _as_int(_get("photo_hold_sec", 6), 6)),
            feed_watch_sec=max(0, _as_int(_get("feed_watch_sec", 150), 150)),
            extra_block_keywords=[str(k) for k in _as_list(_get("extra_block_keywords", []))],
            enable_memory_writeback=_as_bool(_get("enable_memory_writeback", True), True),
            dry_run=_as_bool(_get("dry_run", False), False),
        )


# ---------------------------------------------------------------- 数据结构

@dataclass
class FoodInfo:
    name: str = "看不清"
    category: str = "未知"
    edible: bool = True
    danger: list = field(default_factory=list)
    appearance: str = ""
    confidence: float = 0.0

    def as_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "category": self.category,
                "edible": self.edible,
                "danger": self.danger,
                "appearance": self.appearance,
            },
            ensure_ascii=False,
        )

    def danger_text(self) -> str:
        return "、".join(str(d) for d in self.danger if d) or "不适合吃的东西"


@dataclass
class FeedDecision:
    decision: str = "eat"          # eat | refuse
    emotion: str = ""
    reply_text: str = ""
    reason: str = ""
    eat_scene_prompt: str = ""
    forced: str = ""               # cooldown | satiety | daily_limit | ""
    llm_failed: bool = False


# ---------------------------------------------------------------- 安全硬闸

def precheck_images(paths: list[str], min_bytes: int) -> str:
    """纯代码预检：返回拒绝原因键（命中即拒绝），无问题返回空串。"""
    for path in paths:
        try:
            if not os.path.isfile(path):
                continue
            size = os.path.getsize(path)
        except Exception:
            continue
        if size <= 0:
            return "image_lost"
        if min_bytes and size < min_bytes:
            return "blocked_small"
        if not _decodable(path):
            return "unreadable"
    return ""


def _decodable(path: str) -> bool:
    """能解码就返回 True；PIL 不可用时不做判断（放行给视觉模型）。"""
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return True
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def keyword_blocked(text: str, keywords: list) -> bool:
    if not keywords:
        return False
    lowered = str(text or "").lower()
    return any(str(k).strip().lower() in lowered for k in keywords if str(k).strip())


# ---------------------------------------------------------------- 状态

def today_key(now: Optional[float] = None) -> str:
    return datetime.fromtimestamp(now if now is not None else time.time()).strftime("%Y-%m-%d")


def state_path(data_dir: str) -> str:
    return os.path.join(data_dir, STATE_FILE_NAME)


def load_states(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_states(path: str, states: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(states, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception as e:
        logger.debug(f"bot_treat: 状态写入失败: {e}")


def prune_states(states: dict, today: str) -> dict:
    """跨天重置：day 不是今天的记录清空计数，只保留 last_ts 供冷却使用。"""
    for uid, st in list(states.items()):
        if not isinstance(st, dict):
            states.pop(uid, None)
            continue
        if str(st.get("day") or "") != today:
            st["day"] = today
            st["accepted"] = 0
            st["total"] = 0
            st["last_ts"] = 0.0
    return states


def get_state(states: dict, uid: str, today: str) -> dict:
    st = states.get(uid)
    if not isinstance(st, dict):
        st = {"day": today, "accepted": 0, "total": 0, "last_ts": 0.0}
        states[uid] = st
    st.setdefault("day", today)
    st.setdefault("accepted", 0)
    st.setdefault("total", 0)
    st.setdefault("last_ts", 0.0)
    return st


def note_feed(states: dict, uid: str, today: str, *, accepted: bool, now: Optional[float] = None) -> None:
    st = get_state(states, uid, today)
    st["day"] = today
    st["total"] = int(st.get("total") or 0) + 1
    if accepted:
        st["accepted"] = int(st.get("accepted") or 0) + 1
    st["last_ts"] = float(now if now is not None else time.time())


def forced_reason(state: dict, cfg: TreatConfig, now: Optional[float] = None) -> str:
    """必须拒绝的情境原因（空串=没有强制约束）。"""
    now = float(now if now is not None else time.time())
    accepted = int(state.get("accepted") or 0)
    last_ts = float(state.get("last_ts") or 0.0)
    if int(state.get("total") or 0) > 0 and cfg.cooldown_sec > 0 and (now - last_ts) < cfg.cooldown_sec:
        return "cooldown"
    if accepted >= cfg.satiety_threshold:
        return "satiety"
    if accepted >= cfg.daily_limit:
        return "daily_limit"
    return ""


def since_last_text(state: dict, now: Optional[float] = None) -> str:
    last_ts = float(state.get("last_ts") or 0.0)
    if last_ts <= 0:
        return "还没喂过"
    delta = max(0.0, float(now if now is not None else time.time()) - last_ts)
    if delta < 60:
        return f"{int(delta)} 秒前"
    if delta < 3600:
        return f"{int(delta // 60)} 分钟前"
    return f"{int(delta // 3600)} 小时前"


# ---------------------------------------------------------------- 食物识别

def _clamp_confidence(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return 0.0
    if num > 1.0:
        num = num / 100.0
    return min(1.0, max(0.0, num))


def _clean_danger(value: Any) -> list:
    items = value if isinstance(value, list) else ([value] if value else [])
    return [re.sub(r"\s+", "", str(x))[:12] for x in items if str(x).strip()][:4]


async def recognize_food(bridge, paths: list[str], cfg: TreatConfig) -> Optional[FoodInfo]:
    """视觉识别；返回 None 表示识别失败（不代表不是食物）。"""
    raw = await bridge.vision(
        FOOD_RECOGNITION_SYSTEM,
        paths,
        preferred_provider_id=cfg.vision_provider_id,
        max_tokens=400,
        timeout=float(cfg.vision_timeout_sec),
    )
    obj = _flatten_json(raw or "")
    if not obj:
        return None
    name = str(obj.get("name") or "").strip() or "看不清"
    info = FoodInfo(
        name=name[:24],
        category=str(obj.get("category") or "未知").strip()[:12],
        edible=_as_bool(obj.get("edible"), True),
        danger=_clean_danger(obj.get("danger")),
        appearance=str(obj.get("appearance") or "").strip()[:60],
        confidence=_clamp_confidence(obj.get("confidence")),
    )
    # 明显不是食物 / 认不出的，统一按"不可吃"走硬闸
    if not info.edible and not info.danger:
        info.danger = ["非食物"]
    if info.name in ("看不清", "未知", "无法识别") and info.confidence < 0.3:
        info.edible = False
        if not info.danger:
            info.danger = ["非食物"]
    return info


# ---------------------------------------------------------------- 吃不吃决策

def _fallback_decision(forced: str, food: FoodInfo, *, llm_failed: bool = True) -> FeedDecision:
    if forced:
        key = {"cooldown": "cooldown", "satiety": "satiety", "daily_limit": "quota_exhausted"}[forced]
        return FeedDecision(
            decision="refuse",
            emotion="吃不消",
            reply_text=FALLBACK_TEXTS[key],
            reason={"cooldown": "刚吃过", "satiety": "太饱了", "daily_limit": "今日到量"}[forced],
            forced=forced,
            llm_failed=llm_failed,
        )
    return FeedDecision(
        decision="eat",
        emotion="勉强",
        reply_text=FALLBACK_TEXTS["decide_failed_eat"],
        reason="给面子",
        eat_scene_prompt=f"正在吃{food.name}",
        llm_failed=llm_failed,
    )


async def decide_eat(
    bridge,
    food: FoodInfo,
    state: dict,
    cfg: TreatConfig,
    persona: str,
    forced: str,
    now: Optional[float] = None,
) -> FeedDecision:
    now = float(now if now is not None else time.time())
    today_count = int(state.get("total") or 0)
    satiety = int(state.get("accepted") or 0)

    forced_note = ""
    if forced == "cooldown":
        forced_note = "（你刚刚才吃完，这次必须拒绝，理由围绕「刚吃过 / 还没消化」）"
    elif forced == "satiety":
        forced_note = "（你今天已经吃得够多了，必须拒绝，理由围绕「太饱了」）"
    elif forced == "daily_limit":
        forced_note = "（今天的投喂量已经到上限，必须拒绝，理由围绕「吃不下了 / 今天够了」）"

    user_prompt = build_decision_user(
        food_json=food.as_json(),
        today_count=today_count,
        since_last_text=since_last_text(state, now),
        satiety=satiety,
        threshold=cfg.satiety_threshold,
        limit_note=f"（每日上限 {cfg.daily_limit} 份）",
        forced_note=forced_note,
    )
    raw = await bridge.llm(
        user_prompt,
        system_prompt=build_decision_system(persona),
        max_tokens=500,
        task="feed_decision",
        timeout=float(cfg.llm_timeout_sec),
    )
    obj = _flatten_json(raw or "")
    if not obj:
        return _fallback_decision(forced, food, llm_failed=True)

    decision = str(obj.get("decision") or "").strip().lower()
    if decision not in ("eat", "refuse"):
        decision = "refuse" if forced else "eat"
    dec = FeedDecision(
        decision=decision,
        emotion=re.sub(r"\s+", "", str(obj.get("emotion") or ""))[:12],
        reply_text=re.sub(r"\s+", " ", str(obj.get("reply_text") or "")).strip()[:120],
        reason=re.sub(r"\s+", " ", str(obj.get("reason") or "")).strip()[:60],
        eat_scene_prompt=re.sub(r"\s+", " ", str(obj.get("eat_scene_prompt") or "")).strip()[:160],
    )
    # 代码强制层压过模型：情境不允许吃就一律拒绝
    if forced:
        dec.decision = "refuse"
        dec.forced = forced
        if not dec.reply_text:
            dec.reply_text = FALLBACK_TEXTS[
                {"cooldown": "cooldown", "satiety": "satiety", "daily_limit": "quota_exhausted"}[forced]
            ]
    if not dec.reply_text:
        dec.reply_text = FALLBACK_TEXTS["gen_failed"] if dec.decision == "eat" else "唔…这次就先不吃了。"
    if dec.decision == "eat" and not dec.eat_scene_prompt:
        dec.eat_scene_prompt = f"正在开心地吃{food.name}"
    if dec.decision == "refuse" and not dec.reason:
        dec.reason = "不太想吃"
    return dec
