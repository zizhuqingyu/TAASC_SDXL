"""Structured prompt and vision-language-agent helpers for TA-ASC."""

from __future__ import annotations

import importlib
import math
from typing import Dict, Iterable, List


AGENT_SCORE_ALIASES = {
    "entities": ("entities", "rentities"),
    "attributes": ("attributes",),
    "relations": ("relations", "raletions"),
}


def validate_sample(data: dict) -> None:
    if not isinstance(data, dict):
        raise TypeError("Each JSONL record must be an object")
    required = ("id", "caption", "entities", "attributes", "relations")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Prompt sample is missing fields: {', '.join(missing)}")
    if not isinstance(data['caption'], str) or not data['caption'].strip():
        raise ValueError("caption must be a non-empty string")
    image_id = data['id']
    if isinstance(image_id, bool) or not isinstance(image_id, (str, int)):
        raise ValueError("id must be a string or integer; quote ids to preserve leading zeros")
    if not str(image_id).strip() or str(image_id) in ('.', '..') or any(c in str(image_id) for c in ('/', '\\', '\x00')):
        raise ValueError("Invalid image id")
    for group in ("entities", "attributes", "relations"):
        if not isinstance(data[group], list):
            raise TypeError(f"Sample field {group!r} must be a list.")
    entity_ids = set()
    for entity in data['entities']:
        if not isinstance(entity, dict) or not isinstance(entity.get('label'), str) or not entity['label'].strip():
            raise ValueError("Each entity needs a non-empty label")
        eid = entity.get('id')
        if isinstance(eid, bool) or not isinstance(eid, (str, int)) or eid in entity_ids:
            raise ValueError("Entity ids must be unique strings/integers")
        entity_ids.add(eid)
    for group, phrase_field, references in (
        ('attributes', 'value', ('entity_id',)),
        ('relations', 'predicate', ('subject_id', 'object_id')),
    ):
        for item in data[group]:
            if not isinstance(item, dict) or not isinstance(item.get(phrase_field), str) or not item[phrase_field].strip():
                raise ValueError(f"Each {group} item needs a non-empty {phrase_field}")
            if any(item.get(field) not in entity_ids for field in references):
                raise ValueError(f"Unknown entity reference in {group}: {item}")


def get_agent_score_group(agent_result: dict, group_name: str) -> list:
    if not agent_result:
        return []
    for key in AGENT_SCORE_ALIASES[group_name]:
        if key in agent_result:
            return agent_result[key]
    return []


def parse_agent_score(group_name: str, index: int, raw_score) -> float:
    if isinstance(raw_score, bool):
        raise ValueError(
            f"Invalid boolean score at {group_name}[{index}]: {raw_score!r}"
        )
    try:
        score = float(raw_score)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Non-numeric score at {group_name}[{index}]: {raw_score!r}"
        ) from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(
            f"Score must be finite and in [0, 1] at "
            f"{group_name}[{index}]: {raw_score!r}"
        )
    return score


def validate_agent_result(data: dict, agent_result: dict) -> None:
    if not isinstance(agent_result, dict):
        raise TypeError(
            f"Agent result must be a dict, got {type(agent_result).__name__}."
        )
    for group_name in ("entities", "attributes", "relations"):
        scores = get_agent_score_group(agent_result, group_name)
        expected = data.get(group_name, [])
        if not isinstance(scores, list):
            raise TypeError(f"Agent field {group_name!r} must be a list.")
        if len(scores) != len(expected):
            raise ValueError(
                f"Agent score count mismatch for {group_name}: "
                f"expected {len(expected)}, got {len(scores)}."
            )
        for index, raw_score in enumerate(scores):
            parse_agent_score(group_name, index, raw_score)


def load_agent_module(module_name: str):
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"Could not import agent module {module_name!r}. Put the module on "
            "PYTHONPATH or run with --disable_agent for a single-pass smoke test."
        ) from error
    if not callable(getattr(module, "client_para", None)):
        raise AttributeError(
            f"Agent module {module_name!r} must expose client_para(image_path, annotation)."
        )
    return module


def call_agent_with_retry(
    agent_module,
    image_path,
    annotation: dict,
    data: dict,
    response_attempts: int = 2,
) -> dict:
    last_error = None
    if response_attempts < 1:
        raise ValueError("response_attempts must be positive")
    for attempt in range(1, response_attempts + 1):
        try:
            result = agent_module.client_para(str(image_path), annotation)
            validate_agent_result(data, result)
        except Exception as error:
            last_error = error
            if attempt < response_attempts:
                print(f"Agent request {attempt}/{response_attempts} failed; retrying: {error}", flush=True)
                continue
            break
        print(f"Agent response {attempt}/{response_attempts}: {result}", flush=True)
        return result
    raise ValueError(
        f"The agent failed {response_attempts} times: {last_error}"
    ) from last_error


def scores_all_reach_threshold(agent_result: dict, threshold: float) -> bool:
    for group_name in ("entities", "attributes", "relations"):
        scores = get_agent_score_group(agent_result, group_name)
        for index, score in enumerate(scores):
            if parse_agent_score(group_name, index, score) < threshold:
                return False
    return True


def build_agent_result(data: dict, agent_result: dict) -> Dict[str, float]:
    """Map phrases to their minimum agent confidence."""

    groups = (
        (data["entities"], get_agent_score_group(agent_result, "entities"), "label"),
        (
            data["attributes"],
            get_agent_score_group(agent_result, "attributes"),
            "value",
        ),
        (
            data["relations"],
            get_agent_score_group(agent_result, "relations"),
            "predicate",
        ),
    )
    phrase_scores: Dict[str, float] = {}
    for items, scores, phrase_field in groups:
        if len(items) != len(scores):
            raise ValueError(
                f"Data/agent count mismatch for phrase field {phrase_field!r}."
            )
        for index, (item, score) in enumerate(zip(items, scores)):
            phrase = str(item[phrase_field])
            numeric = parse_agent_score(phrase_field, index, score)
            phrase_scores[phrase] = min(
                phrase_scores.get(phrase, float("inf")),
                numeric,
            )
    return phrase_scores


def build_annotation(data: dict) -> dict:
    entity_map = {
        entity["id"]: entity["label"]
        for entity in data["entities"]
    }
    return {
        "entities": [entity["label"] for entity in data["entities"]],
        "attributes": [
            [entity_map[item["entity_id"]], item["value"]]
            for item in data["attributes"]
        ],
        "relations": [
            [
                entity_map[item["subject_id"]],
                item["predicate"],
                entity_map[item["object_id"]],
            ]
            for item in data["relations"]
        ],
    }


def get_prompt_phrases(data: dict) -> List[str]:
    phrases: Iterable[str] = (
        [str(item["label"]) for item in data["entities"]]
        + [str(item["value"]) for item in data["attributes"]]
        + [str(item["predicate"]) for item in data["relations"]]
    )
    return list(dict.fromkeys(phrases))


def build_attention_feedback_targets(
    *,
    data: dict,
    agent_result: dict | None,
    position_info: Dict[str, dict],
    score_threshold: float,
    max_gain: float,
) -> List[dict]:
    """Convert phrase confidences into Eq. (15) attention gains."""

    if not agent_result:
        return []
    groups = (
        ("entities", data.get("entities", []), "label", "entity"),
        ("attributes", data.get("attributes", []), "value", "attribute"),
        ("relations", data.get("relations", []), "predicate", "relation"),
    )
    targets = []
    for group_name, items, phrase_field, kind in groups:
        scores = get_agent_score_group(agent_result, group_name)
        if len(scores) != len(items):
            raise ValueError(
                f"Agent score count mismatch for {group_name}: "
                f"expected {len(items)}, got {len(scores)}."
            )
        for index, (item, raw_score) in enumerate(zip(items, scores)):
            phrase = str(item[phrase_field])
            info = position_info.get(phrase)
            if info is None:
                info = next(
                    (
                        value
                        for key, value in position_info.items()
                        if str(key).casefold() == phrase.casefold()
                    ),
                    None,
                )
            if info is None:
                print(
                    f"[TA-ASC attention] prompt tokens not found for "
                    f"{kind}: {phrase!r}"
                )
                continue

            confidence = parse_agent_score(group_name, index, raw_score)
            deficit = max(0.0, float(score_threshold) - confidence)
            if deficit <= 0.0:
                continue
            gain = float(max_gain) * deficit / max(float(score_threshold), 1e-8)
            targets.append(
                {
                    "key": f"{kind}:{index}:{phrase}",
                    "phrase": phrase,
                    "kind": kind,
                    "token_positions": info["positions"],
                    "confidence": confidence,
                    "gain": gain,
                }
            )
    return targets


def update_lambda_value_dict(
    phrase_scores: Dict[str, float],
    lambda_values: Dict[str, float],
    *,
    target_score: float = 0.8,
    eta: float = 0.25,
    max_step: float = 0.05,
    lambda_max: float = 0.9,
) -> Dict[str, float]:
    """Apply Eq. (7): bounded confidence-deficiency compensation."""

    updated = {}
    for phrase, current_lambda in lambda_values.items():
        if phrase not in phrase_scores:
            updated[phrase] = current_lambda
            continue
        score = max(0.0, min(1.0, float(phrase_scores[phrase])))
        deficiency = max(0.0, target_score - score)
        delta = min(eta * deficiency, max_step)
        updated[phrase] = round(
            min(float(current_lambda) + delta, lambda_max),
            4,
        )
    return updated
