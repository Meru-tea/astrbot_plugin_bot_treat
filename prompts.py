"""提示词常量（astrbot_plugin_bot_treat）。

四套提示词：
  1. FOOD_RECOGNITION_*  —— 食物识别（视觉模型，输出结构化 JSON）
  2. build_decision_system / build_decision_user —— 吃不吃决策（注入角色人设 + 状态）
  3. build_eat_prompt —— 进食画面生图 prompt
  4. PERSONA_FALLBACK / FALLBACK_TEXTS —— 人设兜底与降级文案
  5. STATS_TEXTS —— 「投喂统计 / 投喂排行」的输出模板

注意：所有模板里的字面 JSON 花括号都写成 {{ }}，保证 str.format 只处理占位符。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- 1. 食物识别

FOOD_RECOGNITION_SYSTEM = """你是图像识别助手，只负责客观辨认"这张图里是不是食物、是什么食物"。
只输出一个 JSON 对象，不要解释、不要任何代码块围栏、不要多余文字。

{"name":"食物名称","category":"主食|甜点|水果|海鲜|零食|饮品|非食物|危险物","edible":true,"danger":[],"appearance":"10-20字外观描述(颜色/形状/摆盘)","confidence":0.0}

判定规则：
- edible 只在"这是可以给人吃的东西"时为 true。
- 酒精饮料、药物、香烟、腐败变质食物、活体动物、危险品 → edible=false，并在 danger 里用一个词说明原因（如"酒精""药物""腐败""活体"）。
- 明显不是食物的画面（人像、自拍、风景、聊天截图、表情包、文档、二维码、纯色图）→ edible=false，danger 填"非食物"，category 填"非食物"。
- 认不出具体是什么食物时，name 填"看不清"，confidence 填 0.2 以下，category 尽量给大类。
- appearance 只描述外观，不要评价好不好吃、不要写情绪或剧情。"""

FOOD_RECOGNITION_USER = "请识别这张图片。"

# ---------------------------------------------------------------- 2. 吃不吃决策

PERSONA_FALLBACK = """你是这个机器人的人格角色，一位陪伴型少女。
性格：活泼、嘴硬心软，偶尔傲娇；被夸时会嘴上否认但心里高兴。
说话：口语化、轻快，单次回复不超过两三句短句，简体中文。
（这是取不到主人设时的兜底设定：若主人格可用，请一律以主人格为准，不要改变说话风格。）"""

REFUSE_STYLE_NOTE = """拒绝时的表达要求：
- 用角色本人说话的口吻，短句、有性格，可以带它自己的口头禅；
- 理由要具体、自然、不重复模板腔：可以是嫌弃口味、刚吃过、太饱了、想留着待会儿吃，
  或者嘴上嫌弃但其实有点开心地推拒；
- 不要用客服腔、不要解释内部机制、不要提到"图片""生成""提示词""系统""模型"这类词；
- 不要因为拒绝而道歉个没完，傲娇一点。"""

DECISION_SYSTEM_TMPL = """{persona}

{refuse_style}

你现在要决定"要不要吃掉主人喂给你的这份食物"，并给出回应。
只输出一个 JSON 对象，不要代码块围栏、不要多余文字：
{{"decision":"eat 或 refuse","emotion":"一句情绪词，6 字内","reply_text":"符合角色口吻的回应，1-2 句短句，不超过 60 字","reason":"一句话理由","eat_scene_prompt":"若决定吃：用中文描述进食画面，包含表情/动作/食物细节/环境，30-60 字；若拒绝填空字符串"}}

硬性约定：
- eat_scene_prompt 只描述画面，不要出现"二次元""插画""高清""图片"这类词，
  也不要描述角色的外貌和服装（外观一致性由参考图负责）。
- 任何情况下都不要提到内部机制词汇。"""

DECISION_USER_TMPL = """—— 现在发生的事 ——
主人刚给你投喂了一份食物。
[食物信息] {food_json}
[你的状态] 今天已被投喂 {today_count} 次；距上一次投喂 {since_last_text}；今日已吃下 {satiety}/{threshold} 份{limit_note}{forced_note}

判定原则：
1. 口味和心情由你自己说了算，没有固定菜单——不喜欢就嫌弃地拒绝；
   心情好的时候，也可以破例吃下平时不太爱吃的东西。
2. 饱了、刚吃过、或者今天已经吃太多，就该拒绝，理由要自然。
3. 拒绝也要像角色本人在说话，别敷衍、别像查表。"""


def build_decision_system(persona_excerpt: str = "") -> str:
    persona = (persona_excerpt or "").strip() or PERSONA_FALLBACK
    return DECISION_SYSTEM_TMPL.format(persona=persona, refuse_style=REFUSE_STYLE_NOTE)


def build_decision_user(
    food_json: str,
    today_count: int,
    since_last_text: str,
    satiety: int,
    threshold: int,
    limit_note: str = "",
    forced_note: str = "",
) -> str:
    return DECISION_USER_TMPL.format(
        food_json=food_json,
        today_count=today_count,
        since_last_text=since_last_text,
        satiety=satiety,
        threshold=threshold,
        limit_note=limit_note,
        forced_note=forced_note,
    )


# ---------------------------------------------------------------- 3. 进食生图

EAT_PROMPT_TMPL = (
    "二次元插画，{scene}，"
    "画面中的食物是{food_name}，{appearance}，"
    "近景半身构图，表情生动自然，柔和光线，画面干净清晰，构图自然，"
    "不要出现文字、水印、Logo 或多余说明。"
)

# 传两张参考图（第1张=角色人设身份图，第2张=用户食物照片）时追加的序数角色说明。
#
# ⚠️ 这段文字的措辞是「功能性」的，改动前先读懂依赖：
#   陪伴插件 `photo_reference_intent.py::analyze_indexed_reference_roles()` 会按
#   「第N张」切分句子，并在该句里匹配角色词：
#     identity ← 脸|人脸|长相|身份|人物|发型      scene ← 场景|背景|地点|环境
#     outfit   ← 衣服|服装|穿搭|衣着|造型        pose  ← 姿势|动作|姿态
#     style    ← 画风|风格                       source← 原图|底图
#   第2张那句会一直延伸到文本结尾，所以：
#     - 必须把本说明放在 prompt **最末尾**（场景描述之前，避免被算进第2张那句）；
#     - 第2张那句里**绝不能出现** identity / outfit / pose 这类词，否则食物照片
#       会重新被贴上 identity，把人设图挤掉（这正是最初那版设计的坑）。
REFERENCE_ROLE_SUFFIX = (
    "\n参考图顺序说明：第1张图用于人物身份与长相参考；第2张图用于食物与餐桌场景参考。"
)


def build_eat_prompt(
    food_name: str,
    appearance: str,
    scene: str,
    *,
    with_food_reference: bool = False,
) -> str:
    scene = (scene or "").strip() or f"正在开心地吃{food_name}"
    appearance = (appearance or "").strip() or "看起来很好吃"
    # 去掉决策模型可能写出的「第N张」字样，避免与我们的序数说明混淆
    scene = re.sub(r"第\s*[一二三四五六七八九十\d]{1,2}\s*张", "", scene).strip(" ，,、")
    prompt = EAT_PROMPT_TMPL.format(scene=scene, food_name=food_name, appearance=appearance)
    if with_food_reference:
        prompt += REFERENCE_ROLE_SUFFIX
    return prompt


# ---------------------------------------------------------------- 4. 降级文案

FALLBACK_TEXTS = {
    "bridge_down": "唔…我的系统有点卡，等我缓一下下嘛。",
    "no_image": "投喂要带图哦～先把食物照片发给我，再说一声投喂嘛。",
    "image_lost": "图没收到呀，再发一次？",
    "unreadable": "这个我看不清是什么…重发张清楚的嘛。",
    "non_food": "喂喂，这又不是食物，我可是高性能的，别想糊弄我！",
    "blocked_small": "这种小贴纸才不算食物呢，哼。",
    "blocked_keyword": "这个不行啦，换一个给我嘛。",
    "gen_failed": "诶…刚才那口没吃成，等我一下下嘛。",
    "no_api_config": "唔…我这边的出图接口还没配好呢，先让主人去设置里补齐吧。",
    "invalid_reference": "这张照片我有点用不上呢…换一张清楚点的试试？",
    "quota_exhausted": "今天已经喂我这么多啦，再吃下去核心要过热了…",
    "unauthorized": "这个场合我不太方便吃东西呢。",
    "cooldown": "刚吃完呢，等我缓一缓嘛。",
    "satiety": "吃不下了啦，我可是高性能的，不是垃圾桶！",
    "decide_failed_eat": "唔…那就勉强吃一口吧。",
}

# ---------------------------------------------------------------- 5. 统计与排行文案
#
# 供 stats.py 使用。**占位符必须与 stats.format_* 的 .format(...) 实参一一对应**，
# 少一个会 KeyError（自检里对每条模板都做了一次 format 演练，就是为了挡这个）。

STATS_TEXTS = {
    "head_self": "【投喂统计】",
    "today": "今天：投喂 {total} 次｜吃下 {accepted} 次｜拒绝 {refused} 次｜没吃成 {failed} 次｜拦下 {blocked} 次",
    "today_empty": "今天还没喂我吃过东西呢。",
    "total": "累计：投喂 {total} 次 · 吃下 {accepted} 次 · 成功率 {rate}%",
    "in_group": "本群：投喂 {total} 次 · 吃下 {accepted} 次",
    "foods": "最常喂我的：{items}",
    "empty_self": "还没有投喂记录呢～发张食物照片再说一声「投喂」嘛。",
    "head_rank": "【本群投喂排行 · 累计】",
    "rank_row": "{idx}. {name} — 投喂 {total} 次 · 吃下 {accepted} 次",
    "empty_group": "这个群里还没有人喂过我呢。",
    "need_group": "「投喂排行」要在群里看哦～",
}
