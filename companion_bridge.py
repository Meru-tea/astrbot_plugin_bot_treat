"""陪伴插件能力桥（astrbot_plugin_atri_treat）。

与 astrbot_plugin_idea_forge/companion_bridge.py 同款防御式写法：
不 import 陪伴插件任何内部模块、不修改其任何文件，全部经 getattr 探测 +
try/except 降级；任一段取不到都自动跳过，本插件不会因此崩溃。

核验过的能力坐标（private_companion 6.5.6）：
  - _persist_private_inbound_images(event, user_id)  private_image.py:321  (async -> list[str] 本地路径)
  - _llm_call(prompt, max_tokens, provider_id, task, *, system_prompt, timeout_seconds, ...)
                                                      token_budget.py:1714
  - _pc_generate_photo_impl(event, prompt, kind, reference_image_paths, caption, send, ...)
                                                      llm_tool_actions.py:3341 (async -> JSON str)
  - _memory_companion_bridge() -> bridge.record_persona_life(...)
                                                      memory_companion_adapter.py:3113 模式
  - 人设文本：插件 config 的 persona_conversation_voice_prompt / reply_style_prompt
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import shutil
import time
import uuid
from typing import Any, Iterable, Optional

from astrbot.api import logger

COMPANION_PLUGIN_NAME = "astrbot_plugin_private_companion"
PLUGIN_INSTANCE_CACHE_SEC = 15.0
PERSONA_KEY = "persona_conversation_voice_prompt"
REPLY_STYLE_KEY = "reply_style_prompt"
# 参考图暂存子目录（建在陪伴插件 data_dir 内，否则会被它的路径白名单拒绝）
REF_STAGING_SUBDIR = "atri_treat_refs"
# 生图接口临时覆盖：plugin 实例 -> 真正的原始端点队列（用于嵌套覆盖时正确还原）
_ENDPOINT_ORIGINALS: dict[int, list] = {}

_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


async def _acall(fn, *args, **kwargs) -> Any:
    """调用可能是 sync/async 的方法并统一等待结果。"""
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _clip(text: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def _flatten_json(text: str) -> dict:
    """从模型输出里尽力抠出第一个 JSON 对象（容忍代码块围栏与前后废话）。"""
    raw = str(text or "").strip()
    if not raw:
        return {}
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _file_uri_to_path(value: str) -> str:
    """把 file:// URI 转成本地路径（兼容 Windows 的 file:///D:/... 形式）。"""
    from urllib.parse import unquote, urlparse

    path = unquote(urlparse(str(value or "")).path)
    if re.match(r"^/[A-Za-z]:/", path):  # Windows: /D:/x -> D:/x
        path = path[1:]
    return path


class CompanionBridge:
    def __init__(self, context):
        self.context = context
        self._plugin: Any = None
        self._found_at = 0.0

    # -------------------------------------------------- 插件实例发现

    def get_plugin(self, force: bool = False) -> Any:
        if (
            not force
            and self._plugin is not None
            and (time.time() - self._found_at) < PLUGIN_INSTANCE_CACHE_SEC
        ):
            return self._plugin
        self._found_at = time.time()
        plugin = None
        try:
            meta = self.context.get_registered_star(COMPANION_PLUGIN_NAME)
            if meta is not None and getattr(meta, "activated", False):
                plugin = getattr(meta, "star_cls", None)
        except Exception as e:
            logger.debug(f"atri_treat: 发现陪伴插件失败: {e}")
            plugin = None
        self._plugin = plugin
        return plugin

    def available(self) -> bool:
        return self.get_plugin() is not None

    # -------------------------------------------------- 身份

    def canonical_user_id(self, event) -> str:
        plugin = self.get_plugin()
        fn = getattr(plugin, "_private_user_id_for_event", None) if plugin else None
        raw = ""
        try:
            raw = str(event.get_sender_id() or "")
        except Exception:
            raw = ""
        if callable(fn):
            try:
                scoped = _clip(fn(event, raw), 160)
                if scoped:
                    return scoped
            except Exception:
                pass
        return raw or "unknown"

    # -------------------------------------------------- 取图

    def _component_images(self, event) -> list[str]:
        """从事件消息链收集图片候选（**含被引用消息里的图**）：本地路径优先，其次 URL。"""
        try:
            chain = getattr(getattr(event, "message_obj", None), "message", None) or []
        except Exception:
            chain = []

        def _name(obj) -> str:
            if isinstance(obj, dict):
                return str(obj.get("type") or "").lower()
            return obj.__class__.__name__.lower()

        def _src(comp) -> str:
            for attr in ("path", "file", "url"):
                value = comp.get(attr) if isinstance(comp, dict) else getattr(comp, attr, None)
                value = str(value or "").strip()
                if not value:
                    continue
                if value.startswith("file://"):
                    value = _file_uri_to_path(value)
                if os.path.isfile(value) or value.lower().startswith("http"):
                    return value
            return ""

        def _walk(comps, depth: int = 0) -> list[str]:
            found: list[str] = []
            if depth > 2:
                return found
            for comp in comps or []:
                name = _name(comp)
                if name == "image":
                    value = _src(comp)
                    if value:
                        found.append(value)
                elif name == "reply":
                    # 引用消息：正文在 comp.chain 里（可能含图）
                    sub_chain = comp.get("chain") if isinstance(comp, dict) else getattr(comp, "chain", None)
                    if sub_chain:
                        found.extend(_walk(sub_chain, depth + 1))
            return found

        return _walk(chain)

    async def _collect_reference_candidates(self, event, user_id: str) -> list[str]:
        """按优先级收集本轮可用的图片源（原始值，可能是本地路径或 URL）。

        七级阶梯，逐级兜底；任何一级失败都不影响后续：
          1. 陪伴插件落盘管道 `_persist_private_inbound_images`
          2. 事件上的 `private_companion_delayed_image_sources`
          3. 陪伴插件本轮图源 `_photo_reference_sources_from_current_event`
          4. 陪伴插件引用图缓存 `_photo_reference_sources_from_reply_cache`（同步）
          5. 陪伴插件引用图事件解析 `_photo_reference_sources_from_reply_event`
          6. 自行遍历消息链（**含引用消息 chain 里的图**）
        """
        plugin = self.get_plugin()
        found: list[str] = []

        def _add_many(values) -> None:
            for value in values or []:
                text = str(value or "").strip()
                if text and text not in found:
                    found.append(text)

        # 1) 落盘管道
        fn = getattr(plugin, "_persist_private_inbound_images", None) if plugin else None
        if callable(fn):
            try:
                _add_many(await _acall(fn, event, user_id))
            except Exception as e:
                logger.debug(f"atri_treat: 落盘管道跳过: {_clip(e, 160)}")

        # 2) 延迟图源
        if not found:
            try:
                _add_many(getattr(event, "private_companion_delayed_image_sources", None))
            except Exception:
                pass

        # 3) 陪伴插件本轮图源（内部还会再跑一次落盘 + raw 提取）
        if not found:
            fn = getattr(plugin, "_photo_reference_sources_from_current_event", None) if plugin else None
            if callable(fn):
                try:
                    _add_many(await _acall(fn, event, user_id))
                except Exception as e:
                    logger.debug(f"atri_treat: 本轮图源解析跳过: {_clip(e, 160)}")

        # 4) 引用图缓存（同步）
        if not found:
            fn = getattr(plugin, "_photo_reference_sources_from_reply_cache", None) if plugin else None
            if callable(fn):
                try:
                    _add_many(await _acall(fn, event))
                except Exception as e:
                    logger.debug(f"atri_treat: 引用图缓存跳过: {_clip(e, 160)}")

        # 5) 引用图事件解析（会走平台接口取被引用消息）
        if not found:
            fn = getattr(plugin, "_photo_reference_sources_from_reply_event", None) if plugin else None
            if callable(fn):
                try:
                    _add_many(await _acall(fn, event))
                except Exception as e:
                    logger.debug(f"atri_treat: 引用图解析跳过: {_clip(e, 160)}")

        # 6) 自行遍历消息链（含引用 chain）
        _add_many(self._component_images(event))
        return found

    async def persist_images(self, event, user_id: str) -> list[str]:
        """把本轮消息里的图片落到本地并返回**本地绝对路径**列表（多路兜底）。"""
        raw = await self._collect_reference_candidates(event, user_id)
        paths: list[str] = []
        for candidate in raw:
            local = await self._local_reference(candidate)
            if local and local not in paths:
                paths.append(local)
        if raw and not paths:
            logger.info(
                f"atri_treat: 发现 {len(raw)} 个图片源但均无法本地化，"
                f"首个={_clip(raw[0], 120)}"
            )
        return paths[:3]

    async def _download(self, url: str) -> str:
        """兜底下载（aiohttp 优先，失败退线程 urllib）。

        落点很关键：陪伴插件对参考图有**路径白名单**——
        `command_handlers.py::_photo_reference_copy_local_file()` 里
        `_photo_reference_path_within_data_dir()` 要求参考图必须位于
        **陪伴插件自己的 data_dir** 之内，否则会报
        「参考图越权本地路径已拒绝」→ 生图回执 `invalid_reference`。
        所以下载的图要落到它的数据目录里（专用子目录，不冒充它的入站图）。
        """
        target_dir = self._reference_staging_dir()
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception:
            return ""
        self._prune_staging_dir(target_dir)
        suffix = os.path.splitext(url.split("?")[0])[1].lower()
        if suffix not in _IMAGE_EXT:
            suffix = ".jpg"
        target = os.path.join(target_dir, f"{int(time.time() * 1000)}{suffix}")
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return ""
                    data = await resp.read()
            if not data:
                return ""
            with open(target, "wb") as fh:
                fh.write(data)
            return target
        except Exception as e:
            logger.debug(f"atri_treat: aiohttp 下载失败，改走 urllib: {e}")
        try:
            import asyncio
            import urllib.request

            def _fetch() -> bytes:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
                    return resp.read()

            data = await asyncio.get_running_loop().run_in_executor(None, _fetch)
            if not data:
                return ""
            with open(target, "wb") as fh:
                fh.write(data)
            return target
        except Exception as e:
            logger.warning(f"atri_treat: 图片下载失败: {e}")
            return ""

    def _reference_staging_dir(self) -> str:
        """参考图暂存目录：优先落在陪伴插件数据目录内（否则会被它的白名单拒绝）。"""
        plugin = self.get_plugin()
        data_dir = str(getattr(plugin, "data_dir", "") or "") if plugin else ""
        if data_dir and os.path.isdir(data_dir):
            return os.path.join(data_dir, REF_STAGING_SUBDIR)
        # 兜底：陪伴插件数据目录不可用时只能落自己这里（随后会被它拒，至少不炸）
        logger.warning("atri_treat: 陪伴插件数据目录不可用，参考图暂存退化为本插件目录")
        return os.path.join(self._cache_dir(), "treat_inbound")

    def _ensure_inside_companion_data_dir(self, plugin, path: str, stem: str = "ref") -> str:
        """确保参考图位于陪伴插件 data_dir 内——不在就复制进去（否则会被其白名单拒绝）。

        为什么：`command_handlers.py::_photo_reference_copy_local_file()` 里的
        `_photo_reference_path_within_data_dir()` 要求参考图必须位于**它自己的 data_dir** 之内，
        越权路径会直接回 `参考图越权本地路径已拒绝` → 生图回执 `invalid_reference`。
        实测踩过：把图下载到本插件目录后就被拒了。
        """
        try:
            data_dir = str(getattr(plugin, "data_dir", "") or "")
            real = os.path.realpath(path)
            root = os.path.realpath(data_dir) if data_dir else ""
        except Exception:
            return ""
        if root and (real == root or real.startswith(root + os.sep)):
            return real  # 已在允许范围内，直接用
        if not root or not os.path.isdir(root):
            return ""
        stage = self._reference_staging_dir()
        try:
            os.makedirs(stage, exist_ok=True)
        except Exception:
            return ""
        suffix = os.path.splitext(real)[1].lower() or ".jpg"
        if suffix not in _IMAGE_EXT:
            suffix = ".jpg"
        target = os.path.join(stage, f"{stem}_{int(time.time() * 1000)}{suffix}")
        try:
            shutil.copy2(real, target)
        except Exception as e:
            logger.warning(f"atri_treat: 复制参考图到陪伴插件数据目录失败: {_clip(e, 160)}")
            return ""
        return target

    @staticmethod
    def _prune_staging_dir(target_dir: str, max_age_sec: float = 86400.0, keep: int = 20) -> None:
        """清理暂存目录：删除超过 1 天的文件，并最多保留 keep 个最新文件。"""
        try:
            entries = [
                os.path.join(target_dir, name)
                for name in os.listdir(target_dir)
                if os.path.splitext(name)[1].lower() in _IMAGE_EXT
            ]
        except Exception:
            return
        now = time.time()
        alive: list[tuple[float, str]] = []
        for path in entries:
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                continue
            if now - mtime > max_age_sec:
                try:
                    os.remove(path)
                except Exception:
                    pass
                continue
            alive.append((mtime, path))
        alive.sort(reverse=True)
        for _, path in alive[max(1, int(keep)):]:
            try:
                os.remove(path)
            except Exception:
                pass

    def _cache_dir(self) -> str:
        try:
            from astrbot.api.star import StarTools

            return str(StarTools.get_data_dir("astrbot_plugin_atri_treat"))
        except Exception:
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    # -------------------------------------------------- 视觉识别

    def _vision_providers(self, preferred_id: str = "") -> Iterable[Any]:
        seen: set[int] = set()
        candidates: list[str] = []
        preferred = str(preferred_id or "").strip()
        if preferred:
            candidates.append(preferred)
        plugin = self.get_plugin()
        for attr in ("PLUGIN_VISION_PROVIDER_ID", "plugin_vision_provider_id"):
            value = getattr(plugin, attr, None) if plugin else None
            if value:
                candidates.append(str(value))
        getter = getattr(self.context, "get_provider_by_id", None)
        for pid in candidates:
            try:
                provider = getter(pid) if callable(getter) else None
            except Exception:
                provider = None
            if provider is not None and id(provider) not in seen:
                seen.add(id(provider))
                yield provider
        try:
            provider = self.context.get_using_provider()
        except Exception:
            provider = None
        if provider is not None and id(provider) not in seen:
            seen.add(id(provider))
            yield provider

    async def vision(
        self,
        prompt: str,
        image_urls: list[str],
        *,
        preferred_provider_id: str = "",
        max_tokens: int = 400,
        timeout: float = 30.0,
    ) -> Optional[str]:
        """视觉理解：图片 + 提示词 → 文本。逐个 provider 尝试，全失败返回 None。"""
        if not image_urls:
            return None
        payloads = self._image_payloads(image_urls)
        last_error = ""
        for provider in self._vision_providers(preferred_provider_id):
            call = getattr(provider, "text_chat", None)
            if not callable(call):
                continue
            for images in payloads:
                try:
                    import asyncio

                    resp = await asyncio.wait_for(
                        call(prompt=prompt, image_urls=images, max_tokens=max_tokens),
                        timeout=max(5.0, float(timeout)),
                    )
                except Exception as e:
                    last_error = str(e)
                    continue
                text = str(getattr(resp, "completion_text", "") or "").strip()
                if not text:
                    chain = getattr(resp, "result_chain", None)
                    text = str(chain or "").strip()
                if text:
                    return text
        if last_error:
            logger.warning(f"atri_treat: 视觉识别全部 provider 失败: {_clip(last_error, 200)}")
        return None

    @staticmethod
    def _image_payloads(image_urls: list[str]) -> list[list[str]]:
        """先按原始值试一遍，再退化为 base64 data URL（部分 provider 不认本地路径）。"""
        first = [str(p) for p in image_urls if p]
        payloads = [first]
        data_urls: list[str] = []
        for path in first:
            if path.lower().startswith("http") or path.startswith("data:"):
                continue
            try:
                import base64

                with open(path, "rb") as fh:
                    blob = fh.read()
                if not blob:
                    continue
                ext = os.path.splitext(path)[1].lower().lstrip(".") or "jpeg"
                if ext == "jpg":
                    ext = "jpeg"
                data_urls.append(f"data:image/{ext};base64,{base64.b64encode(blob).decode()}")
            except Exception:
                continue
        if data_urls and len(data_urls) == len(first):
            payloads.append(data_urls)
        return payloads

    # -------------------------------------------------- 文本 LLM

    async def llm(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        max_tokens: int = 600,
        task: str = "feed_decision",
        timeout: float = 45.0,
        preferred_provider_id: str = "",
    ) -> Optional[str]:
        """优先陪伴插件 _llm_call（带其 token 预算与 provider 路由），失败退默认 provider。"""
        plugin = self.get_plugin()
        if plugin is not None:
            fn = getattr(plugin, "_llm_call", None)
            if callable(fn):
                kwargs: dict = {"max_tokens": max_tokens, "task": task}
                if system_prompt:
                    kwargs["system_prompt"] = system_prompt
                if timeout:
                    kwargs["timeout_seconds"] = float(timeout)
                if preferred_provider_id:
                    kwargs["provider_id"] = preferred_provider_id
                try:
                    import asyncio

                    text = await asyncio.wait_for(
                        fn(prompt, **kwargs), timeout=max(5.0, float(timeout)) + 5.0
                    )
                    if text:
                        return str(text).strip()
                except TypeError:
                    try:
                        import asyncio

                        text = await asyncio.wait_for(
                            fn(prompt, max_tokens=max_tokens),
                            timeout=max(5.0, float(timeout)) + 5.0,
                        )
                        if text:
                            return str(text).strip()
                    except Exception as e:
                        logger.warning(f"atri_treat: 陪伴插件 LLM 调用失败(精简): {_clip(e, 200)}")
                except Exception as e:
                    logger.warning(f"atri_treat: 陪伴插件 LLM 调用失败: {_clip(e, 200)}")
        try:
            import asyncio

            provider = self.context.get_using_provider()
            if provider is not None and callable(getattr(provider, "text_chat", None)):
                resp = await asyncio.wait_for(
                    provider.text_chat(prompt=prompt, session_id=None),
                    timeout=max(5.0, float(timeout)),
                )
                text = getattr(resp, "completion_text", None)
                if text:
                    return str(text).strip()
        except Exception as e:
            logger.warning(f"atri_treat: 兜底 LLM 调用失败: {_clip(e, 200)}")
        return None

    # -------------------------------------------------- 生图接口地址临时覆盖

    def _push_image_endpoint_override(
        self,
        plugin,
        *,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        size: str = "",
    ) -> Optional[dict]:
        """把陪伴插件的「在线生图 API 队列」第一条临时换成给定地址（只改内存）。

        依据（2026-09-26 源码核实）：
        - 运行时真正生效的是**实例属性** `plugin.external_image_api_endpoints`
          （有序队列，取第一条 enabled 的；日志里的 `external_queue_items=1:主在线 API/auto/ready` 即此项）。
          读取处均为 `getattr(self, "external_image_api_endpoints", [])`，不是每次读配置文件。
        - 陪伴插件自己切换 API 的官方命令也是直接改这个属性
          （`command_handlers.py:327` `self.external_image_api_endpoints = changed`）。

        所以投喂这次想换地址，只需临时改属性、用完还原，**不必动配置文件**。
        返回还原所需快照；任何一步失败都返回 None（调用方据此跳过覆盖）。

        嵌套覆盖（两次投喂在 15s 窗口内重叠）时，用模块级 `_ENDPOINT_ORIGINALS` 记录
        **真正的原始队列**，避免"还原成上一次的覆盖值"这种层级污染。
        """
        try:
            key = id(plugin)
            true_original = _ENDPOINT_ORIGINALS.get(key)
            if true_original is None:
                current = getattr(plugin, "external_image_api_endpoints", None)
                true_original = list(current) if isinstance(current, list) else []
                _ENDPOINT_ORIGINALS[key] = true_original
            items = [dict(x) if isinstance(x, dict) else {} for x in true_original]
            if not items:
                items = [{}]
            first = items[0]
            first["base_url"] = str(base_url or first.get("base_url") or "").strip()
            first["api_key"] = str(api_key or first.get("api_key") or "").strip()
            first["model"] = str(model or first.get("model") or "").strip()
            if size:
                first["size"] = str(size).strip()
            first["enabled"] = True
            first.setdefault("name", "主在线 API")
            first.setdefault("platform", "auto")
            plugin.external_image_api_endpoints = items
            return {"plugin": plugin, "original": true_original, "patched": items}
        except Exception as e:
            logger.warning(f"atri_treat: 生图地址覆盖准备失败: {_clip(e, 200)}")
            return None

    @staticmethod
    def _pop_image_endpoint_override(snapshot: Optional[dict]) -> None:
        """还原（幂等）：只在自己那次覆盖仍然生效时还原，避免踩掉更新的覆盖。

        还原目标始终是**真正的原始队列**（见 `_push` 里的层级说明）。
        """
        if not snapshot:
            return
        plugin = snapshot.get("plugin")
        try:
            if getattr(plugin, "external_image_api_endpoints", None) is snapshot.get("patched"):
                plugin.external_image_api_endpoints = snapshot.get("original")
                _ENDPOINT_ORIGINALS.pop(id(plugin), None)
        except Exception as e:
            logger.warning(f"atri_treat: 生图地址还原失败: {_clip(e, 200)}")

    # -------------------------------------------------- 生图

    async def generate_photo(
        self,
        event,
        *,
        prompt: str,
        kind: str = "selfie",
        reference_image_paths: Optional[list[str]] = None,
        caption: str = "",
        send: bool = True,
        timeout: float = 45.0,
        api_base_url: str = "",
        api_key: str = "",
        api_model: str = "",
        api_size: str = "",
        api_override_sec: float = 15.0,
    ) -> dict:
        """调用陪伴插件生图能力，返回解析后的回执 dict（失败返回 {"status": "unavailable"}）。

        `api_*` 非空时会**临时**把陪伴插件的在线生图 API 队列第一条换成给定地址：
        地址解析发生在生图开始后的前若干毫秒（实测 ~20ms），所以只需覆盖前
        `api_override_sec` 秒（默认 15s），随后自动还原——把对她其它生图的影响窗口压到最小。
        """
        plugin = self.get_plugin()
        if plugin is None:
            return {"status": "bridge_unavailable", "generated": False, "sent": False}

        snapshot = None
        restore_timer = None
        if any(str(x or "").strip() for x in (api_base_url, api_key, api_model, api_size)):
            snapshot = self._push_image_endpoint_override(
                plugin,
                base_url=api_base_url,
                api_key=api_key,
                model=api_model,
                size=api_size,
            )
            if snapshot is not None:
                patched = (snapshot.get("patched") or [{}])[0]
                logger.info(
                    f"atri_treat: 本次生图临时使用自定义接口 "
                    f"base={_clip(patched.get('base_url'), 90)} model={patched.get('model')!r} "
                    f"size={patched.get('size')!r}（{float(api_override_sec):.0f}s 后自动还原）"
                )

                async def _restore_later() -> None:
                    try:
                        await asyncio.sleep(max(1.0, float(api_override_sec)))
                    except Exception:
                        pass
                    finally:
                        self._pop_image_endpoint_override(snapshot)

                try:
                    restore_timer = asyncio.create_task(_restore_later())
                except Exception:
                    restore_timer = None

        try:
            return await self._call_generate_photo(
                plugin,
                event,
                prompt=prompt,
                kind=kind,
                reference_image_paths=reference_image_paths,
                caption=caption,
                send=send,
                timeout=timeout,
            )
        finally:
            # 幂等：正常情况下定时器已在 ~15s 时还原，此处多为空操作；
            # 只有"调用立即失败"这种快路径才会真正在这里还原。
            self._pop_image_endpoint_override(snapshot)

    async def _call_generate_photo(
        self,
        plugin,
        event,
        *,
        prompt: str,
        kind: str,
        reference_image_paths: Optional[list[str]],
        caption: str,
        send: bool,
        timeout: float,
    ) -> dict:
        for name in ("_pc_generate_photo_impl", "pc_generate_photo"):
            fn = getattr(plugin, name, None)
            if not callable(fn):
                continue
            kwargs: dict = {
                "event": event,
                "prompt": prompt,
                "kind": kind,
                "caption": caption,
                "send": send,
            }
            if reference_image_paths:
                kwargs["reference_image_paths"] = list(reference_image_paths)
            try:
                import asyncio

                raw = await asyncio.wait_for(fn(**kwargs), timeout=max(30.0, float(timeout)))
            except TypeError as e:
                logger.debug(f"atri_treat: {name} 签名不匹配，试下一个入口: {_clip(e, 160)}")
                continue
            except asyncio.TimeoutError:
                # 生图（尤其带参考图的在线 API）耗时可达 1-3 分钟，超时属于可预期的失败
                logger.warning(
                    f"atri_treat: {name} 生图超时（>{float(timeout):.0f}s）——"
                    f"可上调配置 photo_timeout_sec 后重试"
                )
                return {"status": "timeout", "generated": False, "sent": False, "message": "generation timeout"}
            except Exception as e:
                # 必须带上异常类型：asyncio.TimeoutError 等 str() 为空，只打 message 会得到空白行
                logger.warning(
                    f"atri_treat: {name} 调用异常({type(e).__name__}): {_clip(e, 200) or '(无消息)'}"
                )
                return {
                    "status": "error",
                    "generated": False,
                    "sent": False,
                    "message": f"{type(e).__name__}: {e}",
                }
            payload = _flatten_json(raw) if isinstance(raw, str) else (raw if isinstance(raw, dict) else {})
            if payload:
                return payload
            return {"status": "error", "generated": False, "sent": False, "message": "回执为空"}
        return {"status": "unavailable", "generated": False, "sent": False}

    # -------------------------------------------------- 人设

    def persona_excerpt(self, limit: int = 900) -> str:
        plugin = self.get_plugin()
        if plugin is None:
            return ""
        config = getattr(plugin, "config", None)
        if config is None:
            return ""
        parts: list[str] = []
        for key, label in ((PERSONA_KEY, ""), (REPLY_STYLE_KEY, "回复长度与节奏：")):
            try:
                value = config.get(key) if hasattr(config, "get") else None
            except Exception:
                value = None
            text = str(value or "").strip()
            if text:
                parts.append(f"{label}{text}" if label else text)
        if not parts:
            return ""
        return _clip("\n".join(parts), limit)

    # -------------------------------------------------- 回看最近收到的图

    async def recent_persisted_images(
        self,
        user_ids: Iterable[str],
        max_age_sec: float = 120.0,
        limit: int = 3,
    ) -> list[str]:
        """回看陪伴插件最近落盘的入站图片（应对"照片与文字分两条消息发"的常见情况）。

        陪伴插件自己的图片管道会把每张入站图存到
        `<companion data_dir>/private_inbound_images/<user_id>/<ts>_<idx>.<ext>`，
        与我们的处理是否被触发无关。所以当「投喂」这条消息本身不带图时，
        可以回看该目录里 max_age_sec 秒内的图。

        只在该用户自己的目录里找（不跨用户扫，避免隐私串号）。
        """
        plugin = self.get_plugin()
        data_dir = str(getattr(plugin, "data_dir", "") or "") if plugin else ""
        if not data_dir:
            return []
        base = os.path.join(data_dir, "private_inbound_images")
        return self._scan_recent_images(base, list(user_ids), max_age_sec, limit)

    @staticmethod
    def _scan_recent_images(
        base: str,
        user_ids: list,
        max_age_sec: float,
        limit: int = 3,
    ) -> list[str]:
        """纯函数：在 base/<sanitized uid>/ 下挑出 max_age_sec 内的图，新→旧。"""
        if not base or not os.path.isdir(base):
            return []
        now = time.time()
        hits: list[tuple[float, str]] = []
        for uid in user_ids:
            name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(uid or ""))
            if not name:
                continue
            target = os.path.join(base, name)
            if not os.path.isdir(target):
                continue
            try:
                entries = os.listdir(target)
            except Exception:
                continue
            for entry in entries:
                path = os.path.join(target, entry)
                if os.path.splitext(entry)[1].lower() not in _IMAGE_EXT:
                    continue
                try:
                    if not os.path.isfile(path):
                        continue
                    age = now - os.path.getmtime(path)
                except Exception:
                    continue
                if 0 <= age <= float(max_age_sec):
                    hits.append((now - age, path))
        hits.sort(key=lambda item: item[0], reverse=True)
        result: list[str] = []
        for _, path in hits:
            if path not in result:
                result.append(os.path.abspath(path))
        return result[: max(1, int(limit))]

    # -------------------------------------------------- 人设参考图

    async def persona_reference_path(self, preferred: str = "") -> str:
        """解析人物身份参考图的**本地绝对路径**（多路兜底，取不到返回空串）。

        为什么需要它：陪伴插件在「本轮带了显式参考图」时**不会**再把 persona 候选
        自动加进参考图计划（`proactive_message.py` 约 16263 行那个分支要求
        `not candidates and not paths`）。所以一旦我们传了食物图，就必须自己把
        人设图一并传进去，否则角色的脸会丢。

        为什么必须是本地绝对路径：`image_companion_bridge.py::_image_reference_paths`
        会对每个参考图做 `os.path.isabs` 校验，传 URL 会直接抛 `reference_path_invalid`。
        因此拿到 URL 时先下载成本地文件。

        `preferred`（插件配置里用户自己指定的图，可为本地路径或 URL）优先；
        它同样必须落在陪伴插件 data_dir 内，否则会被其路径白名单拒绝 → 需要时复制进去。
        """
        plugin = self.get_plugin()
        if plugin is None:
            return ""

        want = str(preferred or "").strip()
        if want:
            local = await self._local_reference(want)
            if local:
                staged = self._ensure_inside_companion_data_dir(plugin, local, stem="persona")
                if staged:
                    logger.info(f"atri_treat: 人设参考图使用配置指定的图 → {_clip(staged, 110)}")
                    return staged
                logger.warning(
                    "atri_treat: 配置的人设参考图无法放入陪伴插件数据目录，"
                    "已回退为自动解析（该图不会被使用）"
                )
            else:
                logger.warning(
                    f"atri_treat: 配置的人设参考图不可用（既不是可读的本地文件也不是可下载的 URL）："
                    f"{_clip(want, 120)}"
                )

        candidates: list[str] = []

        # 1) 插件自带解析器（异步，会按需把远程图落到本地）
        fn = getattr(plugin, "_photo_persona_reference_image_path_async", None)
        if callable(fn):
            try:
                candidates.append(str(await _acall(fn) or "").strip())
            except Exception as e:
                logger.debug(f"atri_treat: 人设图异步解析器失败: {_clip(e, 160)}")
        # 2) 同步 getter
        fn = getattr(plugin, "_photo_persona_reference_image_path", None)
        if callable(fn):
            try:
                candidates.append(str(await _acall(fn) or "").strip())
            except Exception:
                pass
        # 3) 实例属性 / 配置字段
        for holder in (plugin, getattr(plugin, "config", None)):
            if holder is None:
                continue
            value = ""
            try:
                value = str(getattr(holder, "photo_persona_reference_image_path", "") or "").strip()
            except Exception:
                value = ""
            if not value and hasattr(holder, "get"):
                try:
                    value = str(holder.get("photo_persona_reference_image_path") or "").strip()
                except Exception:
                    value = ""
            if value:
                candidates.append(value)
        # 4) 参考图目录里的 persona 条目
        for holder in (plugin, getattr(plugin, "config", None)):
            if holder is None or not hasattr(holder, "get"):
                continue
            try:
                catalog = holder.get("photo_reference_catalog") or []
            except Exception:
                catalog = []
            for item in catalog:
                if isinstance(item, dict) and str(item.get("kind") or "") == "persona":
                    candidates.append(str(item.get("source") or item.get("path") or "").strip())

        for value in candidates:
            local = await self._local_reference(value)
            if local:
                return local
        logger.warning("atri_treat: 未能解析出 Atri 人设参考图的本地路径")
        return ""

    @staticmethod
    def _is_local_file(value: str) -> bool:
        text = str(value or "").strip()
        return bool(text) and os.path.isabs(text) and os.path.isfile(text)

    async def _local_reference(self, value: str) -> str:
        """把参考图统一成本地绝对路径：本地文件直接用，URL 下载后再用。"""
        text = str(value or "").strip()
        if not text:
            return ""
        if text.startswith("file://"):
            text = _file_uri_to_path(text)
        if self._is_local_file(text):
            return os.path.abspath(text)
        if text.lower().startswith(("http://", "https://")):
            saved = await self._download(text)
            if saved and self._is_local_file(saved):
                return os.path.abspath(saved)
        return ""

    # -------------------------------------------------- 记忆回写

    async def memory_writeback(self, content: str, tags: list[str], session_id: str = "") -> bool:
        text = str(content or "").strip()
        if not text or not tags:
            return False
        plugin = self.get_plugin()
        if plugin is None:
            return False
        bridge = None
        try:
            fn = getattr(plugin, "_memory_companion_bridge", None)
            bridge = await _acall(fn) if callable(fn) else None
        except Exception:
            bridge = None
        recorder = getattr(bridge, "record_persona_life", None) if bridge is not None else None
        if not callable(recorder):
            return False
        sid = (session_id or "atri_treat")[:180]
        platform = sid.split(":", 1)[0] if ":" in sid else ""
        key = f"atri_treat_{uuid.uuid4().hex[:12]}"
        full_kwargs = {
            "content": text[:800],
            "scope": "unknown",
            "session_id": sid,
            "platform": platform,
            "message_id": key,
            "memory_id": key,
            "memory_type": "atri_treat_feed",
            "reality_level": "bot_action",
            "sayability": "direct",
            "tags": list(tags)[:8],
            "metadata": {
                "event_type": "atri_treat",
                "action_label": "投喂",
                "text": text[:400],
            },
        }
        try:
            await recorder(**full_kwargs)
            return True
        except TypeError:
            try:
                await recorder(content=text[:800], memory_type="atri_treat_feed",
                               metadata={"event_type": "atri_treat", "tags": list(tags)[:8]})
                return True
            except Exception as e:
                logger.warning(f"atri_treat: 记忆回写失败(精简参数): {_clip(e, 160)}")
                return False
        except Exception as e:
            logger.warning(f"atri_treat: 记忆回写失败: {_clip(e, 160)}")
            return False
