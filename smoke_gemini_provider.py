from __future__ import annotations

import json

from llm_provider import get_llm_client
from model_defaults import model_for_role


def main() -> None:
    client = get_llm_client(provider="gemini", force_new=True)
    response = client.generate(
        "Return exactly one compact JSON object with ok=true and label='mirage'. No markdown.",
        model=model_for_role("entity_gen"),
        json_mode=True,
        temperature=0.0,
        max_tokens=128,
        timeout_sec=60,
        max_attempts=2,
    )
    payload = json.loads(str(response.text or "").strip())
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise SystemExit(f"Unexpected Gemini smoke response: {payload!r}")
    print(f"Gemini smoke passed: model={response.model} payload={payload}")


if __name__ == "__main__":
    main()
