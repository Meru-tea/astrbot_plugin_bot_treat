"""投喂统计与排行（astrbot_plugin_bot_treat）。

**为什么不复用 `feed_state.json`**：那个文件是「当日状态」——`decision.prune_states()`
跨天会把计数清零（只留 `last_ts` 供冷却用），而且它的顶层是**扁平的 `{uid: {...}}`**，
把统计塞进去会被 `prune_states` 当成 uid 记录往里注入字段。

所以统计另起一个文件 `feed_stats.json`，独立加载 / 保存 / 清理，互不影响。
老版本升级时文件不存在 → 自动建空骨架，**不迁移、不报错**。

口径（与 README 一致，改动请同步）：

    total = accepted + refused + failed + blocked

  - accepted：决策为「吃」且**出图成功**
  - refused ：决策为「不吃」（含冷却 / 饱腹 / 上限的强制拒绝）
  - failed  ：决策为「吃」但生图失败 / 超时
  - blocked ：进入决策**之前**就被拦（小图 / 解码失败 / 关键词 / 识别失败 / 非食物）

`foods` 只累计**吃下**的食物名（拒绝的不进偏好榜）。

⚠️ 累计字段（total/accepted/... 与 foods）**永不清理**；只有 `by_day` 明细按
`stats_keep_days` 裁剪。这条与 `feed_state` 的「跨天清零」语义**正好相反**，
自检里有一条专门的断言守着它，别"顺手"改回去。
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Optional

from astrbot.api import logger

from .prompts import STATS_TEXTS

STATS_FILE_NAME = "feed_stats.json"
STATS_VERSION = 1

KIND_ACCEPTED = "accepted"
KIND_REFUSED = "refused"
KIND_FAILED = "failed"
KIND_BLOCKED = "blocked"

# 会独立累加的计数键（total 单独在前，它是四者之和）
COUNTER_KEYS = ("total", KIND_ACCEPTED, KIND_REFUSED, KIND_FAILED, KIND_BLOCKED)


# ---------------------------------------------------------------- 路径与读写

def stats_path(data_dir: str) -> str:
    return os.path.join(str(data_dir or "."), STATS_FILE_NAME)


def blank_stats() -> dict:
    return {"version": STATS_VERSION, "users": {}}


def load_stats(path: str) -> dict:
    """读账本；文件缺失 / 损坏 / 结构不对一律返回空骨架，绝不抛异常。"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return blank_stats()
    if not isinstance(data, dict):
        return blank_stats()
    users = data.get("users")
    if not isinstance(users, dict):
        data["users"] = {}
    data.setdefault("version", STATS_VERSION)
    return data


def save_stats(path: str, data: dict) -> None:
    """原子写（tmp + os.replace），与 decision.save_states 同款。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception as e:
        logger.debug(f"bot_treat: 统计数据写入失败: {e}")


# ---------------------------------------------------------------- 记账

def day_key(now: Optional[float] = None) -> str:
    return datetime.fromtimestamp(now if now is not None else time.time()).strftime("%Y-%m-%d")


def _user_rec(data: dict, uid: str) -> dict:
    """取（必要时懒建）某个用户的账本记录。"""
    users = data.setdefault("users", {})
    uid = str(uid or "").strip() or "unknown"
    rec = users.get(uid)
    if not isinstance(rec, dict):
        rec = {}
        users[uid] = rec
    for key in COUNTER_KEYS:
        rec[key] = int(rec.get(key) or 0)
    if not isinstance(rec.get("foods"), dict):
        rec["foods"] = {}
    if not isinstance(rec.get("by_day"), dict):
        rec["by_day"] = {}
    if not isinstance(rec.get("by_group"), dict):
        rec["by_group"] = {}
    rec["name"] = str(rec.get("name") or "")
    rec["first_ts"] = float(rec.get("first_ts") or 0.0)
    rec["last_ts"] = float(rec.get("last_ts") or 0.0)
    return rec


def _bump(bucket: dict, kind: str) -> None:
    """给一个计数桶 +1（total 恒增；四个分项按 kind 增）。"""
    bucket["total"] = int(bucket.get("total") or 0) + 1
    if kind in COUNTER_KEYS[1:]:
        bucket[kind] = int(bucket.get(kind) or 0) + 1


def note_stat(
    data: dict,
    uid: str,
    *,
    kind: str,
    group_id: str = "",
    name: str = "",
    food: str = "",
    now: Optional[float] = None,
    day: Optional[str] = None,
) -> None:
    """唯一写入入口：登记一次投喂回合的结果。

    `kind` 只认 COUNTER_KEYS[1:]；传了别的一律按 blocked 计（宁可少统计也别污染口径）。
    """
    if not isinstance(data, dict):
        return
    if kind not in COUNTER_KEYS[1:]:
        kind = KIND_BLOCKED
    now = float(now if now is not None else time.time())
    day = str(day or day_key(now))

    rec = _user_rec(data, uid)
    if not rec["first_ts"]:
        rec["first_ts"] = now
    rec["last_ts"] = now
    if str(name or "").strip():
        rec["name"] = str(name).strip()[:32]
    _bump(rec, kind)

    # 按天明细（供「今天」这一段使用；跨天不复位累计值）
    bucket = rec["by_day"].get(day)
    if not isinstance(bucket, dict):
        bucket = {}
        rec["by_day"][day] = bucket
    _bump(bucket, kind)

    # 按群明细（供群排行）
    gid = str(group_id or "").strip()
    if gid:
        gbucket = rec["by_group"].get(gid)
        if not isinstance(gbucket, dict):
            gbucket = {}
            rec["by_group"][gid] = gbucket
        _bump(gbucket, kind)
        gbucket["last_ts"] = now

    if kind == KIND_ACCEPTED and str(food or "").strip():
        key = str(food).strip()[:24]
        rec["foods"][key] = int(rec["foods"].get(key) or 0) + 1


def display_name(rec: dict, uid: str = "") -> str:
    """昵称；没存到就退 id 后 6 位（避免在群里暴露完整 QQ 号）。"""
    name = str((rec or {}).get("name") or "").strip()
    if name:
        return name
    tail = str(uid or "").strip()[-6:]
    return f"…{tail}" if tail else "某人"


# ---------------------------------------------------------------- 查询

def _day_bucket(rec: dict, day: str) -> dict:
    bucket = (rec.get("by_day") or {}).get(day)
    return bucket if isinstance(bucket, dict) else {}


def top_foods(rec: dict, limit: int = 3) -> list:
    """食物 → 「吃下」次数，按次数降序；平局按名字升序（保证结果稳定可测）。"""
    foods = (rec or {}).get("foods") or {}
    items = [
        (str(name), int(count or 0))
        for name, count in foods.items()
        if str(name or "").strip() and int(count or 0) > 0
    ]
    items.sort(key=lambda pair: (-pair[1], pair[0]))
    return items[: max(0, int(limit))]


def group_rank(data: dict, group_id: str, limit: int = 5) -> list:
    """本群排行（按累计投喂次数降序；平局按吃下次数降序，再按 uid 升序）。"""
    gid = str(group_id or "").strip()
    rows: list = []
    if not gid:
        return rows
    for uid, rec in ((data or {}).get("users") or {}).items():
        if not isinstance(rec, dict):
            continue
        bucket = (rec.get("by_group") or {}).get(gid)
        if not isinstance(bucket, dict):
            continue
        total = int(bucket.get("total") or 0)
        if total <= 0:
            continue
        rows.append(
            {
                "uid": str(uid),
                "name": display_name(rec, uid),
                "total": total,
                "accepted": int(bucket.get("accepted") or 0),
                "last_ts": float(bucket.get("last_ts") or 0.0),
            }
        )
    rows.sort(key=lambda row: (-row["total"], -row["accepted"], row["uid"]))
    return rows[: max(0, int(limit))]


def prune_stats(data: dict, keep_days: int = 90, now: Optional[float] = None) -> None:
    """只裁 `by_day` 里的过期日子（ISO 日期串可直接字典序比较）；累计值不动。"""
    keep = max(1, int(keep_days))
    if keep <= 0:
        return
    today = day_key(now)
    # 用「今天 - keep 天」的日期串做下界
    cutoff = datetime.fromtimestamp(
        (now if now is not None else time.time()) - keep * 86400.0
    ).strftime("%Y-%m-%d")
    for rec in ((data or {}).get("users") or {}).values():
        if not isinstance(rec, dict):
            continue
        by_day = rec.get("by_day")
        if not isinstance(by_day, dict):
            continue
        for day in list(by_day.keys()):
            if str(day) < cutoff and str(day) != today:
                by_day.pop(day, None)


# ---------------------------------------------------------------- 文案

def _percent(part: int, whole: int) -> int:
    if whole <= 0:
        return 0
    return int(round(part * 100.0 / whole))


def format_user_stats(
    data: dict,
    uid: str,
    *,
    group_id: str = "",
    limit_foods: int = 3,
    now: Optional[float] = None,
) -> str:
    """「投喂统计」的输出。数据为空时给友好文案，不摆一张全 0 的表。"""
    users = (data or {}).get("users") or {}
    rec = users.get(str(uid or ""))
    if not isinstance(rec, dict) or int(rec.get("total") or 0) <= 0:
        return STATS_TEXTS["empty_self"]

    today = _day_bucket(rec, day_key(now))
    lines = [STATS_TEXTS["head_self"]]
    if int(today.get("total") or 0) > 0:
        lines.append(
            STATS_TEXTS["today"].format(
                total=int(today.get("total") or 0),
                accepted=int(today.get("accepted") or 0),
                refused=int(today.get("refused") or 0),
                failed=int(today.get("failed") or 0),
                blocked=int(today.get("blocked") or 0),
            )
        )
    else:
        lines.append(STATS_TEXTS["today_empty"])

    total = int(rec.get("total") or 0)
    accepted = int(rec.get("accepted") or 0)
    lines.append(
        STATS_TEXTS["total"].format(
            total=total,
            accepted=accepted,
            rate=_percent(accepted, total),
        )
    )

    gid = str(group_id or "").strip()
    if gid:
        bucket = (rec.get("by_group") or {}).get(gid)
        if isinstance(bucket, dict) and int(bucket.get("total") or 0) > 0:
            lines.append(
                STATS_TEXTS["in_group"].format(
                    total=int(bucket.get("total") or 0),
                    accepted=int(bucket.get("accepted") or 0),
                )
            )

    foods = top_foods(rec, limit_foods)
    if foods:
        lines.append(
            STATS_TEXTS["foods"].format(
                items="、".join(f"{name} ×{count}" for name, count in foods)
            )
        )
    return "\n".join(lines)


def format_group_rank(
    data: dict,
    group_id: str,
    *,
    limit: int = 5,
    now: Optional[float] = None,
) -> str:
    """「投喂排行」的输出（群内）。"""
    gid = str(group_id or "").strip()
    if not gid:
        return STATS_TEXTS["need_group"]
    rows = group_rank(data, gid, limit)
    if not rows:
        return STATS_TEXTS["empty_group"]
    lines = [STATS_TEXTS["head_rank"]]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            STATS_TEXTS["rank_row"].format(
                idx=idx,
                name=row["name"],
                total=row["total"],
                accepted=row["accepted"],
            )
        )
    return "\n".join(lines)


__all__ = [
    "KIND_ACCEPTED",
    "KIND_BLOCKED",
    "KIND_FAILED",
    "KIND_REFUSED",
    "STATS_FILE_NAME",
    "blank_stats",
    "day_key",
    "display_name",
    "format_group_rank",
    "format_user_stats",
    "group_rank",
    "load_stats",
    "note_stat",
    "prune_stats",
    "save_stats",
    "stats_path",
    "top_foods",
]
