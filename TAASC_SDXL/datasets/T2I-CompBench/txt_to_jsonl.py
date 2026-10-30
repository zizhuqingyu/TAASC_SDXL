#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 caption@@JSON 格式的 TXT 转为带实体实例 ID 的 JSONL。

适用格式：
    caption@@{"entities": ["cube", "cube"],
               "attributes": [["cube", "red"], ["cube", "blue"]],
               "relations": [["cube#1", "on top of", "cube#2"]]}

推荐重复实体直接使用 e1/e2 或 标签#序号 明确指定实例：
    "attributes": [["e1", "red"], ["e2", "blue"]]
    "relations": [["e1", "on top of", "e2"]]

对原始输入中恰好每个重复实体各有一条属性的情况，仅当 caption 中
能逐一定位同名实体与属性且顺序吻合时，才按照出现顺序绑定。
关系只有在句中谓词将同名实体提及唯一地分隔开时才自动消歧。
无法可靠消歧的数据将报错，而不是随意绑定到最后一个实体。

用法：
    python txt_to_jsonl.py input.txt output.jsonl
    python txt_to_jsonl.py input.txt output.jsonl --overrides instance_overrides.json
"""

import argparse
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path


def normalize(value):
    """多词标签/属性/关系归一化；caption 不变。"""
    if not isinstance(value, str):
        raise ValueError(f"要求字符串，实际是 {type(value).__name__}: {value!r}")
    return "_".join(value.strip().split())


def positions(text, phrase):
    """在 caption 中按单词边界搜索，不误将 'car' 命中 'cart'。"""
    phrase = phrase.replace("_", " ").strip()
    if not phrase:
        return []
    pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, phrase.split())) + r"(?!\w)"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.I)]


def parse_attribute(entry):
    if isinstance(entry, dict):
        return entry.get("entity_id", entry.get("entity")), entry.get("value")
    if isinstance(entry, list) and len(entry) == 2:
        return entry
    raise ValueError(f"属性应是 [实体, 属性值] 或对象，实际为 {entry!r}")


def parse_relation(entry):
    if isinstance(entry, dict):
        return entry.get("subject_id", entry.get("subject")), entry.get("predicate"), entry.get("object_id",
                                                                                                entry.get("object"))
    if isinstance(entry, list) and len(entry) == 3:
        return entry
    raise ValueError(f"关系应是 [主语, 关系, 宾语] 或对象，实际为 {entry!r}")


def convert_record(caption, data, record_id):
    labels = data.get("entities")
    if not isinstance(labels, list) or not all(isinstance(x, str) and x.strip() for x in labels):
        raise ValueError("entities 必须是非空字符串组成的列表")

    entities = []
    by_label = defaultdict(list)
    for index, raw_label in enumerate(labels, 1):
        eid = f"e{index}"
        entities.append({"id": eid, "label": normalize(raw_label)})
        by_label[normalize(raw_label)].append(eid)
    all_ids = {e["id"] for e in entities}

    def candidates(ref):
        """返回可能指向的实例集合；显式 eN/label#N 始终唯一。"""
        if not isinstance(ref, str):
            raise ValueError(f"实体引用须为字符串，实际为 {ref!r}")
        ref = ref.strip()
        if ref in all_ids:
            return [ref]
        match = re.fullmatch(r"(.+?)#([1-9]\d*)", ref)
        if match:
            label, occurrence = normalize(match.group(1)), int(match.group(2))
            ids = by_label.get(label, [])
            if occurrence > len(ids):
                raise ValueError(f"找不到实体引用 {ref!r}，有效序号为 1 至 {len(ids)}")
            return [ids[occurrence - 1]]
        ids = by_label.get(normalize(ref), [])
        if not ids:
            raise ValueError(f"找不到实体引用 {ref!r}；entities={labels!r}")
        return ids

    raw_attributes = data.get("attributes", [])
    raw_relations = data.get("relations", [])
    if not isinstance(raw_attributes, list) or not isinstance(raw_relations, list):
        raise ValueError("attributes 和 relations 必须是列表")

    # 保留原属性顺序；先识别显式 ID/唯一实体，再处理同名实体属性。
    resolved_attributes = [None] * len(raw_attributes)
    unresolved = defaultdict(list)
    assigned_by_label = defaultdict(set)
    for i, entry in enumerate(raw_attributes):
        ref, value = parse_attribute(entry)
        value = normalize(value)
        ids = candidates(ref)
        if len(ids) == 1:
            resolved_attributes[i] = {"entity_id": ids[0], "value": value}
            label = entities[int(ids[0][1:]) - 1]["label"]
            assigned_by_label[label].add(ids[0])
        else:
            unresolved[normalize(ref)].append((i, value))

    for label, pending in unresolved.items():
        ids = [eid for eid in by_label[label] if eid not in assigned_by_label[label]]
        if len(pending) == 1 and len(ids) == 1:
            i, value = pending[0]
            resolved_attributes[i] = {"entity_id": ids[0], "value": value}
            continue
        if len(pending) != len(ids):
            raise ValueError(
                f"同名实体 {label!r} 有 {len(by_label[label])} 个实例，但 {len(pending)} 条属性无法分配；"
                f"请把属性引用写成 e1/e2 或 {label}#1/{label}#2"
            )

        # 必须有足够数量的实体提及，且属性值在 caption 中按实体出现顺序可被定位。
        # 否则仅凭 [label, value] 无法推出它属于哪个实例。
        mentions = positions(caption, label)
        if len(mentions) != len(by_label[label]):
            raise ValueError(
                f"同名实体 {label!r} 在 entities 中有 {len(by_label[label])} 个，"
                f"但 caption 中仅能找到 {len(mentions)} 次独立提及；"
                "无法保证属性一一对应，请改用显式实例 ID 或 overrides"
            )
        previous_attribute_end = -1
        previous_mention_end = -1
        for (i, value), eid in zip(pending, ids):
            # 找到该实体提及之前、上一实体提及之后的对应属性短语
            entity_index = int(eid[1:]) - 1
            occurrence_index = by_label[label].index(eid)
            mention_start, mention_end = mentions[occurrence_index]
            matches = [p for p in positions(caption, value)
                       if p[0] >= max(previous_attribute_end, previous_mention_end)
                       and p[1] <= mention_start]
            if not matches:
                raise ValueError(
                    f"无法确定同名实体 {label!r} 的属性 {value!r} 属于 {eid}；"
                    "请用显式实例 ID 指定归属"
                )
            previous_attribute_end = matches[0][1]
            previous_mention_end = mention_end
            resolved_attributes[i] = {"entity_id": eid, "value": value}

    def resolve_relation_ref(ref, predicate, role):
        ids = candidates(ref)
        if len(ids) == 1:
            return ids[0]
        label = normalize(ref)
        mention_spans = positions(caption, label)
        predicate_spans = positions(caption, predicate)
        if len(mention_spans) != len(ids) or len(predicate_spans) != 1:
            raise ValueError(
                f"关系 {predicate!r} 中的实体 {label!r} 指向 {len(ids)} 个实例，"
                "无法唯一消歧，请使用 e1/e2 或 标签#序号"
            )
        rel_start, rel_end = predicate_spans[0]
        before = [i for i, (_, end) in enumerate(mention_spans) if end <= rel_start]
        after = [i for i, (start, _) in enumerate(mention_spans) if start >= rel_end]
        # 必须谓词前后恰好各有一个同名实体；不能只取最近的一个而猜错。
        if len(before) == 1 and len(after) == 1:
            return ids[before[0] if role == "subject" else after[0]]
        raise ValueError(
            f"关系 {predicate!r} 中 {label!r} 的 {role} 无法唯一定位；"
            "请显式写 e1/e2 或在 overrides 中指定关系"
        )

    relations = []
    for entry in raw_relations:
        subject, predicate, obj = parse_relation(entry)
        relations.append({
            "subject_id": resolve_relation_ref(subject, predicate, "subject"),
            "predicate": normalize(predicate),
            "object_id": resolve_relation_ref(obj, predicate, "object"),
        })

    return {
        "id": f"{record_id:06d}",
        "caption": caption,
        "entities": entities,
        "attributes": resolved_attributes,
        "relations": relations,
    }


def convert_txt_to_jsonl(input_path, output_path, overrides_path=None):
    input_path, output_path = Path(input_path), Path(output_path)
    overrides = {}
    if overrides_path is not None:
        with Path(overrides_path).open("r", encoding="utf-8-sig") as f:
            overrides = json.load(f)
        if not isinstance(overrides, dict):
            raise ValueError("overrides 文件必须是以 TXT 行号为键的 JSON 对象")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # 先写临时文件，全部成功才替换输出；避免生成有缺失/错误的半成品。
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=output_path.stem + ".", suffix=".tmp", delete=False,
    ) as temp:
        temp_path = Path(temp.name)
        try:
            with input_path.open("r", encoding="utf-8-sig") as fin:
                for line_number, line in enumerate(fin, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        caption, raw_json = line.split("@@", 1)
                        data = json.loads(raw_json)
                        if not isinstance(data, dict):
                            raise ValueError("@@ 后应是 JSON 对象")
                        # overrides 不改实体列表和原始 caption，仅允许修正难以消歧的属性、关系。
                        if str(line_number) in overrides:
                            override = overrides[str(line_number)]
                            if not isinstance(override, dict) or not set(override).issubset(
                                    {"attributes", "relations"}):
                                raise ValueError("overrides 每条只允许包含 attributes / relations")
                            data = {**data, **override}
                        converted = convert_record(caption.strip(), data, count + 1)
                        temp.write(json.dumps(converted, ensure_ascii=False) + "\n")
                        count += 1
                    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                        raise ValueError(f"输入文件第 {line_number} 行转换失败：{exc}") from exc
            temp.flush()
            temp_path.replace(output_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    return count


def main():
    parser = argparse.ArgumentParser(description="TXT -> JSONL，保留重复实体实例并严格匹配属性/关系")
    parser.add_argument("input", nargs="?", default="ccccolor_val_deepseek.txt", help="输入 TXT")
    parser.add_argument("output", nargs="?", default="color_val_deepseek.jsonl", help="输出 JSONL")
    parser.add_argument("--overrides", default=None, help="可选：按原 TXT 行号指定有歧义的属性/关系")
    args = parser.parse_args()
    n = convert_txt_to_jsonl(args.input, args.output, args.overrides)
    print(f"转换完成：{n} 条记录 -> {args.output}")


if __name__ == "__main__":
    main()
