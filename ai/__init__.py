import requests
import base64
import json
import re
from typing import Any, Callable

from openai import OpenAI
from dotenv import load_dotenv
import os

# Načtení proměnných z .env souboru
load_dotenv()

# Nastavení API klíče
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=api_key
)
from typing import Any

# ============================================
# PRICE TRACKING
# ============================================

# OpenAI GPT-4o-mini pricing (USD per 1M tokens)
PRICE_INPUT_PER_1M = 0.150  # $0.150 per 1M input tokens
PRICE_OUTPUT_PER_1M = 0.600  # $0.600 per 1M output tokens
USD_TO_CZK = 23.0  # Exchange rate

# Global usage tracking
_total_input_tokens = 0
_total_output_tokens = 0
_total_cost_czk = 0.0

_SCHEMA_KEYS = {
    "type", "properties", "required", "items", "additionalProperties",
    "oneOf", "anyOf", "allOf", "enum", "const", "title", "description",
    "$schema", "$ref", "schema", "json_schema", "name"
}

def _looks_like_schema(d: dict) -> bool:
    # typický JSON schema objekt
    if "type" in d and any(k in d for k in ("properties", "items", "oneOf", "anyOf", "allOf")):
        return True
    # wrappery kolem schématu
    if "json_schema" in d and isinstance(d.get("json_schema"), dict):
        return True
    if "schema" in d and isinstance(d.get("schema"), dict):
        return True
    # hodně schema-klíčů bez užitečných dat
    schema_key_hits = sum(1 for k in d.keys() if k in _SCHEMA_KEYS)
    return schema_key_hits >= max(3, len(d) // 2)

def normalize_ai_result(obj: Any) -> Any:
    """
    Generic "unwrap" for AI JSON outputs.
    - unwraps schema-like dicts (type/properties/schema/json_schema/...)
    - unwraps single-key wrappers (data/result/output/...)
    - falls back to searching nested values/lists
    Returns the most likely payload (dict/list/primitive).
    Raises ValueError if nothing usable found.
    """
    # 1) list: zkus najít payload uvnitř
    if isinstance(obj, list):
        for item in obj:
            try:
                return normalize_ai_result(item)
            except ValueError:
                pass
        raise ValueError("No usable payload found in list")

    # 2) dict: unwrap schema-like
    if isinstance(obj, dict):
        # common schema wrappers
        if "json_schema" in obj and isinstance(obj["json_schema"], dict):
            return normalize_ai_result(obj["json_schema"])
        if "schema" in obj and isinstance(obj["schema"], dict):
            return normalize_ai_result(obj["schema"])

        # JSON schema object itself
        if _looks_like_schema(obj):
            if "properties" in obj and isinstance(obj["properties"], dict):
                return normalize_ai_result(obj["properties"])
            if "items" in obj:
                return normalize_ai_result(obj["items"])

        # 3) single-key wrapper → unwrap (typicky {"result": {...}})
        if len(obj) == 1:
            (only_val,) = obj.values()
            return normalize_ai_result(only_val)

        # 4) pokud to NEvypadá jako schema, ber to jako payload
        if not _looks_like_schema(obj):
            return obj

        # 5) jinak DFS do hodnot
        for v in obj.values():
            try:
                return normalize_ai_result(v)
            except ValueError:
                pass

        raise ValueError("No usable payload found in dict")

    # 6) primitive (string/int/float/bool/None) – může být validní payload
    if obj is None:
        raise ValueError("Payload is None")
    return obj


def _track_usage(input_tokens: int, output_tokens: int) -> float:
    """
    Track token usage and calculate cost.

    Args:
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens used

    Returns:
        Cost in CZK for this call
    """
    global _total_input_tokens, _total_output_tokens, _total_cost_czk

    # Calculate cost in USD
    input_cost_usd = (input_tokens / 1_000_000) * PRICE_INPUT_PER_1M
    output_cost_usd = (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_1M
    total_cost_usd = input_cost_usd + output_cost_usd

    # Convert to CZK
    cost_czk = total_cost_usd * USD_TO_CZK

    # Update global counters
    _total_input_tokens += input_tokens
    _total_output_tokens += output_tokens
    _total_cost_czk += cost_czk

    return cost_czk


def get_price() -> dict:
    """
    Get current total AI usage cost.

    Returns:
        Dictionary with usage statistics and total cost in CZK
    """
    return {
        "input_tokens": _total_input_tokens,
        "output_tokens": _total_output_tokens,
        "total_tokens": _total_input_tokens + _total_output_tokens,
        "cost_czk": _total_cost_czk,
        "cost_czk_formatted": f"{_total_cost_czk:.4f} Kč"
    }


def restart_price():
    """
    Reset the price tracking counters.
    """
    global _total_input_tokens, _total_output_tokens, _total_cost_czk
    _total_input_tokens = 0
    _total_output_tokens = 0
    _total_cost_czk = 0
    print("AI usage counters reset.")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Funkce pro odeslání obrázku a dotazu na OpenAI API
def send_image_and_question(image_path, question, json_schema):
    base64_image = encode_image(image_path)
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "image_response",
                "schema": json_schema,
        },
    }

    }


    response = client.chat.completions.create(**kwargs)

    # Track usage and cost
    if response.usage:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost_czk = _track_usage(input_tokens, output_tokens)
        print(f"💰 AI call cost: {cost_czk:.4f} Kč (in: {input_tokens}, out: {output_tokens}) | Total: {_total_cost_czk:.4f} Kč")

    ret = response.choices[0].message.content

    if ret is None:
        raise ValueError("No response from OpenAI API.")

    return normalize_ai_result(json.loads(ret))


def fix_json_input(malformed_json_string):
    # Odstranění přebytečných čárek před uzavíracími závorkami
    malformed_json_string = re.sub(r',\s*([\]}])', r'\1', malformed_json_string)
    return malformed_json_string

