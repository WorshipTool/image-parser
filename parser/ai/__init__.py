"""
AI module for OpenAI API integration and response normalization.
"""

import base64
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env file
load_dotenv()

# Set API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=api_key
)

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
    """Check if dict looks like a JSON schema object."""
    # Typical JSON schema object
    if "type" in d and any(k in d for k in ("properties", "items", "oneOf", "anyOf", "allOf")):
        return True
    # Wrappers around schema
    if "json_schema" in d and isinstance(d.get("json_schema"), dict):
        return True
    if "schema" in d and isinstance(d.get("schema"), dict):
        return True
    # Many schema keys without useful data
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
    # 1) list: try to find payload inside
    if isinstance(obj, list):
        for item in obj:
            try:
                return normalize_ai_result(item)
            except ValueError:
                pass
        raise ValueError("No usable payload found in list")

    # 2) dict: unwrap schema-like
    if isinstance(obj, dict):
        # Common schema wrappers
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

        # 3) single-key wrapper → unwrap (typically {"result": {...}})
        if len(obj) == 1:
            (only_val,) = obj.values()
            return normalize_ai_result(only_val)

        # 4) if it doesn't look like schema, treat it as payload
        if not _looks_like_schema(obj):
            return obj

        # 5) otherwise DFS into values
        for v in obj.values():
            try:
                return normalize_ai_result(v)
            except ValueError:
                pass

        raise ValueError("No usable payload found in dict")

    # 6) primitive (string/int/float/bool/None) – can be valid payload
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
    """
    Encode image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def send_prompt_with_schema(
    system_prompt: str,
    user_prompt: str,
    json_schema: dict,
    image_path: str = None,
    schema_name: str = "response"
) -> Any:
    """
    Send a prompt to OpenAI with optional image and JSON schema response format.

    Args:
        system_prompt: System instructions (rules, constraints, etc.)
        user_prompt: User message (task description, data to process, etc.)
        json_schema: JSON schema for structured response
        image_path: Optional path to image file
        schema_name: Name for the JSON schema (default: "response")

    Returns:
        Parsed and normalized JSON response
    """
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Build user message
    if image_path:
        base64_image = encode_image(image_path)
        user_content = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    else:
        user_content = user_prompt

    messages.append({
        "role": "user",
        "content": user_content
    })

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": json_schema
            }
        }
    )

    # Track usage and cost
    if response.usage:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        _track_usage(input_tokens, output_tokens)

    ret = response.choices[0].message.content

    if ret is None:
        raise ValueError("No response from OpenAI API.")

    return normalize_ai_result(json.loads(ret))


def fix_json_input(malformed_json_string):
    """
    Fix common JSON formatting issues.

    Removes trailing commas before closing brackets.
    """
    # Remove trailing commas before closing brackets
    malformed_json_string = re.sub(r',\s*([\]}])', r'\1', malformed_json_string)
    return malformed_json_string

