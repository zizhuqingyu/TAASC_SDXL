# -*- coding: utf-8 -*-
"""将每行纯文本 caption 用 DeepSeek 转为完整的 JSONL 记录。

每行格式：{"id":"000001","caption":"...","entities":[],"attributes":[],"relations":[]}
无需命令行参数：python deepseek_shiti_shuxing.py
支持把旧 caption@@JSON 输出自动备份并就地转换为 JSONL；断点续跑、失败单独记录。
"""

import json
import re
import sys
import time
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

# =================== 只需修改这里 ===================
API_KEY = "sk-a13668557a1c4505958933b0a2f76479"                 # 你指定的固定 API Key（须为真实有效密钥）
INPUT_TXT = "ori_ABC-6K.txt"
OUTPUT_TXT = "ABC-6K_deepseek.jsonl"

# 即使输出文件扩展名是 .txt，里面每一行仍是完整 JSON（JSONL 格式）。
OVERWRITE = False                # False: 续跑/自动修复旧输出；True: 备份并重新处理全部
MODEL_NAME = "deepseek-chat"
API_BASE_URL = "https://api.deepseek.com/v1"
MAX_RETRY = 3
SLEEP_TIME = 0.3
# ====================================================

BAD_PREDICATES = {
    "and", "or", "but", "other", "another", "also",
    "with other", "as well as", "plus", "including", "include", "includes",
}

SYSTEM_PROMPT = """You extract VISUALLY GROUNDED entities, properties, and relations from one caption.
Return a SINGLE valid JSON object, without Markdown. Be conservative; never invent information.
Schema:
{
 "entities": [
   {"id":"e1", "label":"goat", "mention":"a black goat", "occurrence":1, "attributes":["black"]},
   {"id":"e2", "label":"goat", "mention":"a white goat", "occurrence":1, "attributes":["white"]}
 ],
 "relations": [
   {"subject_id":"e1", "predicate":"beside", "object_id":"e2"}
 ]
}
RULES:
1. Each DISTINCT named object, including same-named objects, gets a different ID e1, e2, e3...
   Order IDs by the start position of each object's FIRST occurrence in the caption.
   Do not merge two goats, two cars, etc. Never identify relations or attributes by a name.
2. Every entity must have an EXACT source-text mention (a contiguous substring of caption).
   Use a COMPLETE identifying noun phrase, including its local modifiers; e.g., 'a black goat',
   NOT merely 'goat' when the caption also describes a white goat.
   'label' is the noun/noun phrase as it appears in the source mention.
   If the same mention string occurs multiple times, set 'occurrence' = 1, 2, ...
3. Every attribute must refer to its own entity. An attribute must occur literally in the caption.
   For same-label instances, attach attribute evidence to that instance's own mention.
   Do NOT assign black to white goat or vice versa. Preserve source spelling.
4. Relation endpoints MUST be entity IDs (subject_id, object_id), never names or strings like 'goat'.
   Predicate MUST appear verbatim in the caption. No conjunction-only relations.
   A shared relation may involve multiple specific entity IDs only if that scope is clear.
   If relation participants or direction are ambiguous, OMIT the uncertain relation.
5. An explicitly described group (e.g. 'two white sheep') is ONE group entity,
   with attributes ['two', 'white']; do not fabricate individual sheep identities.
6. Don't hallucinate; emit [] for empty lists. No extra keys.

EXAMPLE INPUT:
two white sheep, a black goat and a white goat in a field
EXAMPLE OUTPUT:
{
 "entities": [
  {"id":"e1","label":"sheep","mention":"two white sheep","occurrence":1,"attributes":["two","white"]},
  {"id":"e2","label":"goat","mention":"a black goat","occurrence":1,"attributes":["black"]},
  {"id":"e3","label":"goat","mention":"a white goat","occurrence":1,"attributes":["white"]},
  {"id":"e4","label":"field","mention":"a field","occurrence":1,"attributes":[]}
 ],
 "relations": [
  {"subject_id":"e1","predicate":"in","object_id":"e4"},
  {"subject_id":"e2","predicate":"in","object_id":"e4"},
  {"subject_id":"e3","predicate":"in","object_id":"e4"}
 ]
}
EXAMPLE INPUT:
a red ball beside another red ball
EXAMPLE OUTPUT:
{
 "entities": [
  {"id":"e1","label":"ball","mention":"a red ball","occurrence":1,"attributes":["red"]},
  {"id":"e2","label":"ball","mention":"another red ball","occurrence":1,"attributes":["red"]}
 ],
 "relations": [{"subject_id":"e1","predicate":"beside","object_id":"e2"}]
}
"""


def require_str(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须为非空字符串")
    return value.strip()


def occurrences(text, phrase):
    """精确英文词边界，兼容大小写和多空格，返回 (start,end) 列表。"""
    if not isinstance(phrase, str) or not phrase.strip():
        return []
    pat = r"(?<!\w)" + r"\s+".join(re.escape(x) for x in phrase.strip().split()) + r"(?!\w)"
    return [m.span() for m in re.finditer(pat, text, flags=re.I)]


def label_normalize(s):
    return re.sub(r"\s+", "_", s.strip())


def parse_response(content):
    content = require_str(content, "API 响应")
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.S | re.I)
    if fenced:
        content = fenced.group(1)
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("API 必须返回 JSON 对象")
    return parsed


def validate_and_convert(caption, graph, index):
    """静态校验能检查实例 ID/字符串证据/常见同名属性错绑；无法推断自然语言所有歧义。"""
    raw_ents = graph.get("entities")
    raw_rels = graph.get("relations")
    if not isinstance(raw_ents, list) or not isinstance(raw_rels, list):
        raise ValueError("entities、relations 必须是列表")
    if not raw_ents:
        raise ValueError("这条非空 caption 未抽取到任何实体；不能将可能的漏提取记录当作成功")

    entities, attrs, relations = [], [], []
    anchors, raw_label, raw_attrs = {}, {}, {}
    for pos, ent in enumerate(raw_ents, start=1):
        if not isinstance(ent, dict):
            raise ValueError(f"第 {pos} 个实体不是对象")
        eid = require_str(ent.get("id"), "实体 id")
        if eid != f"e{pos}":
            raise ValueError(f"实体实例 ID 必须按顺序为 e1/e2/...，但第 {pos} 个是 {eid}")
        label = require_str(ent.get("label"), f"{eid}.label")
        mention = require_str(ent.get("mention"), f"{eid}.mention")
        if not occurrences(caption, label):
            raise ValueError(f"{eid} 的 label {label!r} 不在原文中")
        if not occurrences(mention, label):
            raise ValueError(f"{eid} 的 mention {mention!r} 不包含 label {label!r}")
        found = occurrences(caption, mention)
        if not found:
            raise ValueError(f"{eid} 的 mention {mention!r} 不是原文连续片段")
        occ = ent.get("occurrence", 1)
        if type(occ) is not int or not 1 <= occ <= len(found):
            raise ValueError(f"{eid} 的 occurrence 无效：{occ}")
        span = found[occ-1]
        ent_attrs = ent.get("attributes")
        if not isinstance(ent_attrs, list):
            raise ValueError(f"{eid}.attributes 必须是列表")
        clean = []
        for val in ent_attrs:
            val = require_str(val, f"{eid} 的属性")
            if not occurrences(caption, val):
                raise ValueError(f"{eid} 的属性 {val!r} 不在原文中")
            if val.lower() not in {x.lower() for x in clean}:
                clean.append(val)
                attrs.append({"entity_id": eid, "value": label_normalize(val)})
        entities.append({"id": eid, "label": label_normalize(label)})
        raw_label[eid] = label.casefold()
        raw_attrs[eid] = clean
        anchors[eid] = (span, caption[span[0]:span[1]])

    starts = [anchors[f"e{i}"][0][0] for i in range(1, len(entities)+1)]
    if starts != sorted(starts):
        raise ValueError("e1/e2/... 的实体顺序与原文出现顺序不一致")
    counts = Counter(raw_label.values())
    for eid, lab in raw_label.items():
        if counts[lab] <= 1:
            continue
        span, mention = anchors[eid]
        for other_eid, other_lab in raw_label.items():
            if other_eid == eid or other_lab != lab:
                continue
            other_span, _ = anchors[other_eid]
            if max(span[0], other_span[0]) < min(span[1], other_span[1]):
                raise ValueError(f"重复实体 {eid}/{other_eid} 的原文锚点重叠，无法唯一区分")
        for attr in raw_attrs[eid]:
            # 对同名实体采用保守校验：属性需要出现在本实体的完整短语中，
            # 防止两个 goat 的 black/white 交叉归属。共享范围属性不确定时交由人工复核。
            if not occurrences(mention, attr):
                raise ValueError(f"重复实体 {eid} ({mention!r}) 的属性 {attr!r} 不在自身短语内，疑似错绑")

    seen = set()
    for n, rel in enumerate(raw_rels, 1):
        if not isinstance(rel, dict):
            raise ValueError(f"关系 #{n} 不是对象")
        sub = require_str(rel.get("subject_id"), "关系 subject_id")
        pred = require_str(rel.get("predicate"), "关系 predicate")
        obj = require_str(rel.get("object_id"), "关系 object_id")
        if sub not in anchors or obj not in anchors:
            raise ValueError(f"关系 #{n} 引用了不存在的实体 {sub}->{obj}")
        if sub == obj:
            raise ValueError(f"关系 #{n} 两端引用同一实例 {sub}")
        if not occurrences(caption, pred):
            raise ValueError(f"谓词 {pred!r} 不在原文中")
        if pred.casefold() in BAD_PREDICATES:
            raise ValueError(f"关系谓词 {pred!r} 是连接词，不能作为关系")
        key = (sub, pred.casefold(), obj)
        if key not in seen:
            seen.add(key)
            relations.append({"subject_id":sub, "predicate":label_normalize(pred), "object_id":obj})

    return {
        "id": f"{index:06d}",
        "caption": caption,
        "entities": entities,
        "attributes": attrs,
        "relations": relations,
    }


def extract_caption(client, caption, index):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": caption},
    ]
    last_exc = None
    for attempt in range(1, MAX_RETRY + 1):
        response_text = None
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                temperature=0, response_format={"type": "json_object"},
                max_tokens=2400, stream=False,
            )
            response_text = completion.choices[0].message.content
            return validate_and_convert(caption, parse_response(response_text), index)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            last_exc = exc
            status_code = getattr(exc, "status_code", None)
            if status_code in (401, 402, 403):
                raise RuntimeError(f"DeepSeek 鉴权/权限错误 HTTP {status_code}：检查脚本顶部 API_KEY 和额度") from exc
            print(f"  [第 {index} 条] 第 {attempt}/{MAX_RETRY} 次调用/校验失败：{exc}", file=sys.stderr)
            if attempt < MAX_RETRY:
                if isinstance(response_text, str) and response_text.strip():
                    messages = [
                        {"role":"system", "content":SYSTEM_PROMPT},
                        {"role":"user", "content":caption},
                        {"role":"assistant", "content":response_text},
                        {"role":"user", "content":(
                            f"Your previous output failed validation: {exc}. "
                            "Rewrite the COMPLETE JSON object. Ensure each duplicate-named entity "
                            "has a distinct mention and its own attributes and relation IDs."
                        )},
                    ]
                else:
                    messages = [
                        {"role":"system", "content":SYSTEM_PROMPT},
                        {"role":"user", "content":caption},
                    ]
                time.sleep(min(2*attempt, 5))
    raise RuntimeError(f"第 {index} 条重试 {MAX_RETRY} 次仍未得到可校验的结果：{last_exc}")


def read_captions(path):
    items = []
    with path.open("r", encoding="utf-8-sig") as f:
        for n, raw in enumerate(f, 1):
            val = raw.strip()
            if not val:
                continue
            if "@@" in val:
                raise ValueError(f"输入第 {n} 行含 @@，本程序需要纯句子 TXT，不是旧版输出文件")
            items.append(val)
    return items


def to_output_line(rec):
    """无条件输出完整 JSONL，固定字段顺序，不使用 caption@@JSON。"""
    item = {key: rec[key] for key in ("id", "caption", "entities", "attributes", "relations")}
    return json.dumps(item, ensure_ascii=False)


def from_output_line(line):
    """读取已有输出，兼容完整 JSONL 和上一版的 caption@@JSON，返回(记录,格式)。"""
    line = line.strip()
    if not line:
        raise ValueError("空白输出行")
    if line.startswith("{"):
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError("JSONL 行不是 JSON 对象")
        return obj, "jsonl"

    # 历史版本：caption@@{"id":...,"entities":...,"attributes":...,"relations":...}
    caption, sep, part = line.partition("@@")
    if not sep:
        raise ValueError("既不是 JSONL，也不是可迁移的旧 caption@@JSON")
    obj = json.loads(part)
    if not isinstance(obj, dict):
        raise ValueError("@@ 后不是 JSON 对象")
    if "caption" in obj and obj["caption"] != caption:
        raise ValueError("旧输出的两个 caption 不一致")
    obj["caption"] = caption
    return obj, "legacy"


def check_saved_record(rec, captions):
    if not isinstance(rec, dict):
        return False
    idstr = rec.get("id")
    if not isinstance(idstr, str) or not re.fullmatch(r"\d{6,}", idstr):
        return False
    num = int(idstr)
    if not 1 <= num <= len(captions) or rec.get("caption") != captions[num-1]:
        return False
    ents = rec.get("entities")
    attrs = rec.get("attributes")
    rels = rec.get("relations")
    if not (isinstance(ents,list) and isinstance(attrs,list) and isinstance(rels,list)):
        return False
    eids = set()
    for i, ent in enumerate(ents, 1):
        if not isinstance(ent,dict) or ent.get("id") != f"e{i}" or not isinstance(ent.get("label"),str):
            return False
        eids.add(ent["id"])
    for att in attrs:
        if not isinstance(att,dict) or att.get("entity_id") not in eids or not isinstance(att.get("value"),str):
            return False
    for rel in rels:
        if (not isinstance(rel,dict) or rel.get("subject_id") not in eids or
            rel.get("object_id") not in eids or not isinstance(rel.get("predicate"),str)):
            return False
    return True


def back_up(path):
    """任何重写前先保留完整文件；避免覆盖宝贵的 API 结果。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = path.with_name(path.name + ".backup_" + timestamp)
    shutil.copy2(path, candidate)
    print(f"旧输出已备份：{candidate}")
    return candidate


def rewrite_output(path, rec_map):
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("w", encoding="utf-8") as f:
        for idx in sorted(rec_map):
            f.write(to_output_line(rec_map[idx]) + "\n")
    tmp.replace(path)


def load_and_repair_output(path, captions):
    """自动迁移旧 caption@@JSON，支持跳过有效记录、重跑无效记录。"""
    if not path.exists():
        return {}
    valid, bad = {}, []
    migrated = 0
    with path.open("r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            try:
                data, source_format = from_output_line(line)
                if not check_saved_record(data, captions):
                    raise ValueError("记录 ID、caption 或实体/关系结构与当前输入不匹配")
                num = int(data["id"])
                if num in valid:
                    raise ValueError(f"重复记录 ID {num}")
                valid[num] = data
                if source_format == "legacy":
                    migrated += 1
            except Exception as e:
                bad.append((lineno, str(e)))

    if bad or migrated:
        back_up(path)
        if migrated:
            print(f"检测到旧 caption@@JSON 格式：{migrated} 条已转换为完整 JSONL，无须重复调用 API。")
        if bad:
            print(f"检测到 {len(bad)} 条无法迁移的旧/损坏/不匹配记录，将在本次重做。")
            print(f"首条问题位于原输出第 {bad[0][0]} 行：{bad[0][1]}")
        rewrite_output(path, valid)
    return valid


def main():
    source, dest = Path(INPUT_TXT), Path(OUTPUT_TXT)
    if source.resolve() == dest.resolve():
        raise ValueError("输入路径不能和输出路径相同")
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    if not isinstance(API_KEY, str) or not API_KEY.strip():
        raise ValueError("需要在文件顶部写入有效的 API_KEY")
    captions = read_captions(source)
    if not captions:
        raise ValueError("输入文件没有有效句子")
    dest.parent.mkdir(parents=True, exist_ok=True)
    error_path = dest.with_name(dest.name + ".errors.jsonl")
    if OVERWRITE:
        if dest.exists():
            back_up(dest)
        saved = {}
        rewrite_output(dest, saved)
    else:
        saved = load_and_repair_output(dest, captions)
    pending = [i for i in range(1, len(captions)+1) if i not in saved]
    print(f"输入：{source}，句子总数：{len(captions)}")
    print(f"输出：{dest}，旧结果保留：{len(saved)}，本次待处理：{len(pending)}")
    if not pending:
        error_path.unlink(missing_ok=True)
        print("所有记录已完成，无需重新调用 API。")
        return

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai 依赖：pip install -U openai tqdm") from exc
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda values, **kwargs: values

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL, timeout=90)
    errors = []
    # 即使个别句子校验失败，也继续处理其余句子；不会因一行难例中断整个数据集。
    # 成功数据写入输出文件，失败数据写入 .errors.jsonl；下次运行会自动补做遗漏 ID。
    with dest.open("a", encoding="utf-8") as outf:
        try:
            for idx in tqdm(pending, desc="DeepSeek 信息提取"):
                try:
                    record = extract_caption(client, captions[idx-1], idx)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    if "鉴权/权限错误" in str(exc):
                        print(f"\n{exc}。为避免重复失败，程序停止；已有有效输出不受影响。", file=sys.stderr)
                        raise SystemExit(2)
                    errors.append({"id":f"{idx:06d}", "caption":captions[idx-1], "error":str(exc)})
                    print(f"\n第 {idx} 条校验失败，已记录到错误文件并继续下一条。", file=sys.stderr)
                    continue
                outf.write(to_output_line(record) + "\n")
                outf.flush()
                saved[idx] = record
                if SLEEP_TIME:
                    time.sleep(SLEEP_TIME)
        finally:
            # 只修改输出顺序；保留已验证的记录，缺失 ID 下次补做。
            if saved:
                rewrite_output(dest, saved)
            with error_path.open("w", encoding="utf-8") as errf:
                for e in errors:
                    errf.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n处理完成：共 {len(captions)} 条，成功 {len(saved)} 条，尚需人工检查/下次重试 {len(errors)} 条")
    if errors:
        print(f"错误列表：{error_path}。下次直接重运行本脚本即可自动重试缺失 ID。")
    else:
        error_path.unlink(missing_ok=True)
        print("没有遗漏记录。")


if __name__ == "__main__":
    main()
