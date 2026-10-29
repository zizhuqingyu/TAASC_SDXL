"""HTTP adapter for the existing InternVL scoring server."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import requests


DEFAULT_SERVER_URL = os.getenv(
    "INTERNVL_SERVER_URL",
    "http://127.0.0.1:8000/chat",
)


def client_para(
    image_path: str,
    annotation: Dict[str, Any],
    server_url: str = DEFAULT_SERVER_URL,
    timeout: int = 300,
) -> Dict[str, Any]:
    response = requests.post(
        server_url,
        json={"image_path": image_path, "annotation": annotation},
        timeout=timeout,
    )
    try:
        result = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"Agent server returned non-JSON content (HTTP {response.status_code})."
        ) from error
    if not response.ok:
        raise RuntimeError(
            "Agent server request failed: "
            + json.dumps(result, ensure_ascii=False)
        )
    return result
