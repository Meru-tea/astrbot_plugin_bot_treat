"""独立生图（astrbot_plugin_bot_treat 的 standalone 模式）。

不依赖陪伴插件的任何能力，直接打一个 **OpenAI 兼容** 的图片接口。

协议限制（重要）：**只支持同步协议** —— 响应里直接带 `b64_json` 或 `url`。
异步回执型接口（只返回 `{"status":"submitted","task_id":...}`，要再轮询取图）不支持。

2026-09-29 在 `gemai.huchan.cn` + `gpt-image-2.5` 上的实测结论（照此写死，别再猜）：
  - `POST /images/generations`  JSON `{model,prompt,size,n}` → 200，**直接回 b64_json**
    实测 1024x1024 用 62.1s、768x1344 用 24.8s（两个尺寸都成功）
  - `POST /images/edits`  multipart：单张用 `image`、多张用 `image[]`
    → 200，直接回 b64_json（1 张 47.7s / 2 张 69.6s）
  - **不要传 `response_format`**：gpt-image 系一律返回 b64_json，传该参数反而可能 400。

回执格式与 `companion_bridge.generate_photo` 对齐，`main.py` 才能复用同一套判定：
    成功 {"status":"ok","generated":True,"sent":False,"path":<本地路径>,"message":""}
    失败 {"status":"unauthorized|quota_exhausted|timeout|error|no_api_config",
          "generated":False,"sent":False,"message":...}

`sent` 恒为 False：本模块**不发送**，由 `main.py` 把图片与 caption 组成一条消息链
自己 yield 出去（这样 dry_run 天然不会发出图）。
"""
from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import time
import uuid
from typing import Any, Optional

from astrbot.api import logger

from .companion_bridge import _IMAGE_EXT, _clip

# 出图落盘子目录（本插件自己的 data_dir 下）
OUT_SUBDIR = "out"
# 落盘图清理：超过 1 天或超出 keep 张就删
OUT_KEEP = 30
OUT_MAX_AGE_SEC = 86400.0
DEFAULT_SIZE = "1024x1024"

# 状态码 → 回执 status（与桥对齐）
STATUS_OK = "ok"


def normalize_base_url(base_url: str) -> str:
    """接口地址归一：去尾斜杠；缺 `/v1` 就补上。

    容错三种常见写法：
      https://host          → https://host/v1
      https://host/v1/      → https://host/v1
      https://host/v1/images → https://host/v1/images（已含 /images，原样返回）
    """
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/images") or base.endswith("/images/generations") or base.endswith("/images/edits"):
        return base
    if base.endswith("/v1") or "/v1/" in base:
        return base
    return base + "/v1"


def endpoint_for(base_url: str, *, use_edits: bool) -> str:
    """按是否有参考图选端点。"""
    base = normalize_base_url(base_url)
    if not base:
        return ""
    if base.endswith("/images"):
        return base + ("/edits" if use_edits else "/generations")
    return base + ("/images/edits" if use_edits else "/images/generations")


def classify_http_error(code: Any, body: str = "") -> str:
    """HTTP 错误 → 回执 status（纯函数，离线可测）。"""
    text = str(body or "").lower()
    try:
        num = int(code)
    except Exception:
        num = 0
    if num in (401, 403):
        return "unauthorized"
    if num == 429:
        return "quota_exhausted"
    if num == 402:
        return "quota_exhausted"
    if "insufficient" in text or "quota" in text or "balance" in text:
        return "quota_exhausted"
    if "invalid api key" in text or "unauthorized" in text or "no permission" in text:
        return "unauthorized"
    return "error"


def parse_image_response(payload: Any) -> dict:
    """从响应体里抠出图（纯函数，离线可测）。

    返回 {"kind": "b64"|"url"|"async"|"none", "value": str, "error": str}
    - b64  : base64 图片数据
    - url  : 图片直链
    - async: 异步回执（只给了 task_id）——本模块不支持，回报明确原因
    - none : 结构不认识
    """
    obj = payload
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return {"kind": "none", "value": "", "error": "响应不是 JSON"}
    if not isinstance(obj, dict):
        return {"kind": "none", "value": "", "error": "响应不是 JSON 对象"}
    if obj.get("error"):
        err = obj.get("error")
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return {"kind": "none", "value": "", "error": str(msg or "接口返回 error")}
    data = obj.get("data")
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            b64 = str(item.get("b64_json") or "").strip()
            if b64:
                return {"kind": "b64", "value": b64, "error": ""}
            url = str(item.get("url") or "").strip()
            if url:
                return {"kind": "url", "value": url, "error": ""}
    # 异步回执：明确告知不支持，别让它伪装成"生成失败"
    if obj.get("task_id") or obj.get("taskId") or str(obj.get("status") or "").lower() in (
        "submitted",
        "queued",
        "pending",
        "processing",
    ):
        return {
            "kind": "async",
            "value": "",
            "error": "接口返回异步回执（只有 task_id），独立模式只支持同步出图协议",
        }
    if obj.get("image") or obj.get("image_url"):
        value = str(obj.get("image_url") or obj.get("image") or "").strip()
        if value:
            return {"kind": "b64" if len(value) > 512 else "url", "value": value, "error": ""}
    return {"kind": "none", "value": "", "error": "响应里没有图片数据"}


def out_dir(data_dir: str) -> str:
    return os.path.join(str(data_dir or "."), OUT_SUBDIR)


def prune_out_dir(target_dir: str, max_age_sec: float = OUT_MAX_AGE_SEC, keep: int = OUT_KEEP) -> None:
    """清理出图目录：删过期文件，并最多保留 keep 个最新的。"""
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
        if now - mtime > float(max_age_sec):
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


def save_b64(data_dir: str, b64: str, *, suffix: str = ".png") -> str:
    """base64 → 本地 PNG，返回绝对路径（失败返回空串）。"""
    try:
        raw = base64.b64decode(str(b64 or ""), validate=False)
    except Exception as e:
        logger.warning(f"bot_treat: [独立模式] base64 解码失败: {_clip(e, 160)}")
        return ""
    if not raw:
        return ""
    target_dir = out_dir(data_dir)
    try:
        os.makedirs(target_dir, exist_ok=True)
    except Exception as e:
        logger.warning(f"bot_treat: [独立模式] 出图目录创建失败: {_clip(e, 160)}")
        return ""
    prune_out_dir(target_dir)
    target = os.path.join(target_dir, f"{int(time.time() * 1000)}{suffix}")
    try:
        with open(target, "wb") as fh:
            fh.write(raw)
    except Exception as e:
        logger.warning(f"bot_treat: [独立模式] 出图落盘失败: {_clip(e, 160)}")
        return ""
    return os.path.abspath(target)


def build_multipart(fields: dict, files: list, boundary: str) -> tuple[bytes, str]:
    """手工拼 multipart（与已实测通过的探针写法一致，避免 FormData 差异）。

    files: list[(field_name, filename, bytes)]
    """
    chunks: list[bytes] = []
    for key, value in (fields or {}).items():
        chunks.append((f"--{boundary}\r\n").encode())
        chunks.append((f'Content-Disposition: form-data; name="{key}"\r\n\r\n').encode())
        chunks.append(str(value).encode("utf-8") + b"\r\n")
    for name, filename, data in files or []:
        ctype = mimetypes.guess_type(str(filename))[0] or "application/octet-stream"
        chunks.append((f"--{boundary}\r\n").encode())
        chunks.append(
            (f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n').encode()
        )
        chunks.append((f"Content-Type: {ctype}\r\n\r\n").encode())
        chunks.append(bytes(data) + b"\r\n")
    chunks.append((f"--{boundary}--\r\n").encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _read_refs(reference_paths: Optional[list], field: str) -> tuple[list, str]:
    """读参考图文件 → files 列表。返回 (files, 错误信息)。"""
    files: list = []
    for idx, path in enumerate(reference_paths or []):
        text = str(path or "").strip()
        if not text or not os.path.isfile(text):
            return [], f"参考图不存在或不可读: {_clip(text, 120)}"
        try:
            with open(text, "rb") as fh:
                data = fh.read()
        except Exception as e:
            return [], f"参考图读取失败({type(e).__name__}): {_clip(e, 120)}"
        name = os.path.basename(text) or f"ref{idx}.png"
        files.append((field, name, data))
    return files, ""


async def _post(url: str, *, headers: dict, data: bytes, timeout: float) -> tuple[Optional[int], bytes, str]:
    """POST 后返回 (status, body, transport_error)。transport_error 非空表示连不上/超时。"""
    if not url:
        return None, b"", "接口地址为空"
    try:
        import aiohttp  # type: ignore

        client_timeout = aiohttp.ClientTimeout(total=max(10.0, float(timeout)))
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(url, data=data, headers=headers) as resp:
                body = await resp.read()
                return int(resp.status), body, ""
    except asyncio.TimeoutError:
        return None, b"", "timeout"
    except Exception as e:
        # aiohttp 不可用 / 连接失败 → 退 urllib（放进线程，别阻塞事件循环）
        if "aiohttp" not in repr(e):
            logger.debug(f"bot_treat: [独立模式] aiohttp 请求失败，改走 urllib: {_clip(e, 160)}")

    def _sync() -> tuple[Optional[int], bytes, str]:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=max(10.0, float(timeout))) as resp:  # noqa: S310
                return int(resp.status), resp.read(), ""
        except urllib.error.HTTPError as exc:
            detail = b""
            try:
                detail = exc.read()
            except Exception:
                pass
            return int(exc.code), detail, ""
        except Exception as exc:
            return None, b"", f"{type(exc).__name__}: {exc}"

    try:
        return await asyncio.get_running_loop().run_in_executor(None, _sync)
    except Exception as e:
        return None, b"", f"{type(e).__name__}: {e}"


async def _download_to(data_dir: str, url: str, timeout: float) -> str:
    """url 型响应 → 落盘（与 b64 同一目录）。"""
    if not url.lower().startswith(("http://", "https://")):
        return ""
    try:
        import aiohttp  # type: ignore

        client_timeout = aiohttp.ClientTimeout(total=max(10.0, float(timeout)))
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return ""
                raw = await resp.read()
    except Exception:
        def _sync() -> bytes:
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=max(10.0, float(timeout))) as resp:  # noqa: S310
                return resp.read()

        try:
            raw = await asyncio.get_running_loop().run_in_executor(None, _sync)
        except Exception as e:
            logger.warning(f"bot_treat: [独立模式] 出图下载失败: {_clip(e, 160)}")
            return ""
    if not raw:
        return ""
    target_dir = out_dir(data_dir)
    try:
        os.makedirs(target_dir, exist_ok=True)
    except Exception:
        return ""
    suffix = os.path.splitext(str(url).split("?")[0])[1].lower()
    if suffix not in _IMAGE_EXT:
        suffix = ".png"
    target = os.path.join(target_dir, f"{int(time.time() * 1000)}{suffix}")
    try:
        with open(target, "wb") as fh:
            fh.write(raw)
    except Exception:
        return ""
    return os.path.abspath(target)


async def generate_photo(
    data_dir: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    size: str = DEFAULT_SIZE,
    prompt: str,
    reference_paths: Optional[list] = None,
    kind: str = "selfie",
    timeout: float = 240.0,
) -> dict:
    """独立模式的生图主入口。

    有参考图且 `kind != "text2img"` → `/images/edits`（multipart）；
    否则 → `/images/generations`（JSON）。**不发送**，只把图落到 data_dir/out/。
    """
    started = time.time()
    base = normalize_base_url(base_url)
    key = str(api_key or "").strip()
    model_name = str(model or "").strip()
    if not base or not key or not model_name:
        return {
            "status": "no_api_config",
            "generated": False,
            "sent": False,
            "message": "独立模式需要配置 photo_api_base_url / photo_api_key / photo_api_model",
        }

    refs = [str(p) for p in (reference_paths or []) if str(p or "").strip()]
    use_edits = bool(refs) and str(kind or "").lower() != "text2img"
    url = endpoint_for(base_url, use_edits=use_edits)
    headers = {"Authorization": f"Bearer {key}"}
    size_text = str(size or "").strip() or DEFAULT_SIZE

    if use_edits:
        # 多张用 image[]（OpenAI 兼容写法的实测通过形态）；单张用 image。
        # 若服务端不认 image[]，退一次重复 image 字段。
        attempts = ["image[]" if len(refs) > 1 else "image"]
        if len(refs) > 1:
            attempts.append("image")
        last_error = ""
        for field in attempts:
            files, err = _read_refs(refs, field)
            if err:
                return {"status": "invalid_reference", "generated": False, "sent": False,
                        "message": err}
            boundary = "----bot_treat" + uuid.uuid4().hex
            data, ctype = build_multipart(
                {"model": model_name, "prompt": str(prompt or ""), "size": size_text, "n": 1},
                files,
                boundary,
            )
            send_headers = dict(headers)
            send_headers["Content-Type"] = ctype
            status, body, transport = await _post(url, headers=send_headers, data=data, timeout=timeout)
            result = _finish(
                data_dir, status, body, transport, started, url=url,
                size=size_text, model=model_name, kind=kind, tagged=f"edits:{field}:{len(refs)}张",
            )
            result = await _resolve_deferred(data_dir, result, timeout)
            if result["status"] not in ("error",) or field == attempts[-1]:
                return result
            last_error = result.get("message") or ""
            logger.warning(
                f"bot_treat: [独立模式] /images/edits 用 {field} 失败，改试下一种字段: {_clip(last_error, 160)}"
            )
        return {"status": "error", "generated": False, "sent": False, "message": last_error}

    payload = {"model": model_name, "prompt": str(prompt or ""), "size": size_text, "n": 1}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    send_headers = dict(headers)
    send_headers["Content-Type"] = "application/json"
    status, body, transport = await _post(url, headers=send_headers, data=data, timeout=timeout)
    return await _resolve_deferred(
        data_dir,
        _finish(
            data_dir, status, body, transport, started, url=url,
            size=size_text, model=model_name, kind=kind, tagged="generations",
        ),
        timeout,
    )


def _finish(
    data_dir: str,
    status: Optional[int],
    body: bytes,
    transport: str,
    started: float,
    *,
    url: str,
    size: str,
    model: str,
    kind: str,
    tagged: str,
) -> dict:
    """统一收敛成回执（HTTP 层 → 解析层 → 落盘）。"""
    elapsed = time.time() - started
    text = body.decode("utf-8", "replace") if body else ""
    if transport:
        key = "timeout" if transport == "timeout" else "error"
        logger.warning(
            f"bot_treat: [独立模式] 生图请求失败({tagged}) {key}: {_clip(transport, 200)}"
        )
        return {"status": key, "generated": False, "sent": False,
                "message": _clip(transport, 200), "elapsed": round(elapsed, 1)}
    if status is not None and int(status) >= 400:
        key = classify_http_error(status, text)
        logger.warning(
            f"bot_treat: [独立模式] 生图 HTTP {status}({tagged}) → {key}: {_clip(text, 220)}"
        )
        return {"status": key, "generated": False, "sent": False,
                "message": f"HTTP {status}: {_clip(text, 200)}", "elapsed": round(elapsed, 1)}

    parsed = parse_image_response(text)
    if parsed["kind"] == "b64":
        path = save_b64(data_dir, parsed["value"])
    elif parsed["kind"] == "url":
        # 少数中转回直链而不回 b64 —— 这里必须异步下载
        return {"status": "deferred_url", "generated": False, "sent": False,
                "message": "", "url": parsed["value"], "elapsed": round(elapsed, 1),
                "model": model, "size": size, "kind": kind, "tagged": tagged}
    else:
        path = ""
    if not path:
        msg = parsed["error"] or "图片落盘失败"
        logger.warning(f"bot_treat: [独立模式] 生图未成功({tagged}): {msg}")
        return {"status": "error", "generated": False, "sent": False,
                "message": msg, "elapsed": round(elapsed, 1)}
    logger.info(
        f"bot_treat: [独立模式] 生图成功({tagged}) model={model} size={size} kind={kind} "
        f"耗时={elapsed:.1f}s → {_clip(path, 140)}"
    )
    return {"status": STATUS_OK, "generated": True, "sent": False, "path": path,
            "message": "", "elapsed": round(elapsed, 1)}


async def _resolve_deferred(data_dir: str, receipt: dict, timeout: float) -> dict:
    """把 url 型响应下载落盘，转成正常回执。"""
    if receipt.get("status") != "deferred_url":
        return receipt
    url = str(receipt.get("url") or "")
    path = await _download_to(data_dir, url, timeout)
    if not path:
        return {"status": "error", "generated": False, "sent": False,
                "message": f"url 型响应下载失败: {_clip(url, 160)}",
                "elapsed": receipt.get("elapsed")}
    logger.info(
        f"bot_treat: [独立模式] 生图成功(url:{receipt.get('tagged')}) "
        f"model={receipt.get('model')} size={receipt.get('size')} "
        f"耗时={receipt.get('elapsed')}s → {_clip(path, 140)}"
    )
    return {"status": STATUS_OK, "generated": True, "sent": False, "path": path,
            "message": "", "elapsed": receipt.get("elapsed")}
