"""
Final AI-based correction for song sheets

This module handles the final validation and correction of OCR-processed song sheets
using AI. It runs in two steps:
1. OCR cleanup and chord validation (with image)
2. Section structure correction (text-only)
"""

import cv2
import uuid
from pathlib import Path
from ai import send_image_and_question


def _step1_ocr_cleanup(draft_song_text: str, image_path: str) -> dict:
    """
    STEP 1: OCR cleanup and chord validation using the image.

    Args:
        draft_song_text: Draft song sheet text
        image_path: Path to the full song image

    Returns:
        Dictionary with 'title' and 'sheetData' keys
    """
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "sheetData": {"type": "string"}
        },
        "required": ["title", "sheetData"],
        "additionalProperties": False
    }

    prompt = (
        "You are correcting OCR ERRORS in a SONG SHEET.\n"
        "The IMAGE is the ONLY source of truth for text and chords.\n\n"

        "IMPORTANT TITLE RULES:\n"
        "- The song title MUST NOT appear inside any verse, chorus, or section.\n"
        "- If the title text appears again at the beginning of the song body,\n"
        "  it is NOT a verse and MUST be REMOVED from the sheet body.\n"
        "- A verse NEVER starts with the song title.\n"
        "- The title line is NOT part of lyrics, even if OCR placed it inside {1S}.\n\n"

        "SONG SHEET FORMAT:\n"
        "- Lyrics contain inline chords in square brackets ONLY: [C], [Dm7], [F/A]\n"
        "- Square brackets are RESERVED EXCLUSIVELY for chords.\n"
        "- Sections are marked with tags like {1S}, {2S}, {1R}, {I}, {B}, {O}\n\n"

        "CRITICAL RULE ABOUT BRACKETS:\n"
        "- Square brackets [] MUST contain ONLY valid chord symbols.\n"
        "- Lyrics MUST NEVER be inside square brackets.\n"
        "- If a square bracket does NOT form a valid chord, REMOVE the brackets and keep the text.\n"
        "- NEVER leave unclosed or dangling square brackets.\n\n"

        "YOUR TASKS:\n\n"

        "1) EXTRACT SONG TITLE\n"
        "- Read the song title from the IMAGE.\n"
        "- Return it EXACTLY as written (preserve diacritics and casing).\n\n"

        "2) FIX OCR ERRORS IN LYRICS\n"
        "- Correct typos, broken words, wrong letters.\n"
        "- Fix diacritics and casing errors.\n"
        "- Remove random symbols that don't belong to lyrics.\n"
        "- If OCR wrapped lyrics in [] by mistake, REMOVE the brackets.\n"
        "- Do NOT paraphrase or change meaning.\n\n"

        "3) FIX CHORD SYMBOLS\n"
        "- Valid chord pattern: ^[A-Ga-g][#b]?[0-9+]*([/][A-Ga-g][#b]?)?$\n"
        "- Fix obvious chord typos (e.g. Dmi7 → Dm7, Emi → Em).\n"
        "- REMOVE any bracketed content that does NOT match a valid chord.\n"
        "- Do NOT add new chords.\n\n"

        "4) PRESERVE STRUCTURE\n"
        "- Keep ALL section tags EXACTLY AS THEY ARE, except when a hint line requires a section type.\n"
        "- Do NOT reorder lines.\n"
        "- Only fix TEXT, CHORDS, and REMOVE hint-only lines.\n\n"

        "CONSTRAINTS:\n"
        "- Do NOT invent lyrics or chords.\n"
        "- Do NOT add new section tags beyond what is required by hints.\n"
        "- When unsure, keep the original text WITHOUT brackets.\n\n"

        "RETURN ONLY VALID JSON:\n"
        "{\n"
        "  \"title\": \"<exact song title from image>\",\n"
        "  \"sheetData\": \"<corrected text with section tags applied>\"\n"
        "}\n\n"

        "DRAFT SONG SHEET:\n"
        f"{draft_song_text}"
    )

    result = send_image_and_question(image_path, prompt, json_schema=schema)

    # Ensure result is a dict
    if not isinstance(result, dict):
        return {"title": "", "sheetData": draft_song_text}

    return result


def _step2_section_correction(sheetData: str, title: str) -> dict:
    """
    STEP 2: Section structure correction (text-only, no image).

    Args:
        cleaned_sheet_text: OCR-cleaned sheet from step 1

    Returns:
        Dictionary with 'sheetData' key
    """
    # Import here to avoid circular dependency
    from ai import client

    schema = {
        "type": "object",
        "properties": {
            "sheetData": {"type": "string"},
            "title": {"type": "string"}
        },
        "required": ["sheetData", "title"],
        "additionalProperties": False
    }

    prompt = (
        "You are a strict SONG SHEET NORMALIZER.\n"
        "Input is an already-parsed song in a custom format, but it may contain OCR noise.\n"
        "Your job is to make the sheetData CLEAN, CONSISTENT, and MEANINGFUL.\n"
        "Every character must have a purpose; otherwise remove it.\n\n"

        "INPUTS:\n"
        "- title: the song title (already extracted)\n"
        "- sheetData: the song content\n\n"

        "SHEET FORMAT RULES:\n"
        "- Inline chords are in square brackets ONLY: [C], [Dm7], [F/A]\n"
        "- Section tags are in curly braces at the start of a line.\n\n"

        "VALID CHORD RULES (STRICT):\n"
        "- A chord token must look like a real chord.\n"
        "- Allowed characters inside [] are only: A-G a-g 0-9 # b / +\n"
        "- If a bracketed token contains ANY other character (quotes, commas, weird symbols), fix if obvious.\n"
        "- If it cannot be fixed into a valid chord, REMOVE the whole chord token including brackets.\n"
        "- Never output lyrics inside [].\n"
        "- Never output unclosed/dangling brackets.\n\n"

        "NOISE CLEANUP (STRICT):\n"
        "- Remove stray punctuation/quotes/strokes that are not part of words or chords.\n"
        "- Remove isolated garbage tokens like: \"—\", \"--\", \"=\", random commas/quotes, single symbols.\n"
        "- Keep normal punctuation in lyrics only if it clearly belongs (comma, period).\n\n"

        "HINT REMOVAL (IMPORTANT):\n"
        "- The input may contain structural hints like: 'Ref', 'Refrén', 'Chorus', 'Bridge', '1.', '2.', 'Verse', etc.\n"
        "- These hint words MUST NOT remain anywhere in the output.\n"
        "- If a line is only a hint, DELETE that line.\n"
        "- If a hint is embedded inside a lyric line, REMOVE only the hint part and keep the real lyric text.\n\n"
        "- Make sure that not all sections are labeled as verses; use choruses and other tags as needed.\n\n"

        "ALLOWED SECTION TAGS:\n"
        "{1S}, {2S}, {3S}, ...  = verses\n"
        "{1R}, {2R}, ...       = choruses (must contain lyric text)\n"
        "{B}                  = bridge\n"
        "{I}                  = intro (chord-only, start)\n"
        "{M}                  = interlude (chord-only, between sections)\n"
        "{O}                  = outro (chord-only, end)\n\n"

        "SECTION RULES:\n"
        "- A section tag MUST appear ONLY on the FIRST line of its section.\n"
        "- All following lines in the same section MUST NOT start with a tag.\n"
        "- Chord-only blocks are NEVER choruses.\n"
        "- A chorus MUST contain at least one lyric/text line.\n"
        "- Repeated or near-identical lyric blocks are choruses.\n"
        "- All other lyric blocks are verses numbered in order.\n"
        "- Chord-only blocks:\n"
        "  - at the beginning -> {I}\n"
        "  - at the end -> {O}\n"
        "  - between lyric sections -> {M}\n"
        "- Use {B} only if there is a clearly distinct lyrical/musical bridge.\n\n"

        "TITLE RULES:\n"
        "- The title MUST be returned exactly as given in the input field `title`.\n"
        "- The title MUST NOT appear as the first line of {1S} (or any section).\n"
        "- If the title line exists inside sheetData, remove that line from sheetData.\n\n"

        "HARD CONSTRAINTS:\n"
        "- Do NOT reorder lines.\n"
        "- Do NOT paraphrase lyrics.\n"
        "- Do NOT add new chords.\n"
        "- You may remove invalid chords, hint text, and OCR noise.\n"
        "- Preserve the overall song flow.\n\n"

        "OUTPUT (JSON ONLY):\n"
        "{\n"
        "  \"title\": \"<same as input title>\",\n"
        "  \"sheetData\": \"<cleaned and section-corrected sheetData>\"\n"
        "}\n\n"

        "INPUT TITLE:\n"
        f"{title}\n\n"
        "INPUT SHEETDATA:\n"
        f"{sheetData}"
    )

    # Text-only call (no image)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "section_correction",
                "schema": schema
            }
        }
    )

    import json
    from ai import _track_usage, _total_cost_czk

    # Track usage and cost
    if response.usage:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost_czk = _track_usage(input_tokens, output_tokens)
        print(f"💰 AI call cost: {cost_czk:.4f} Kč (in: {input_tokens}, out: {output_tokens}) | Total: {_total_cost_czk:.4f} Kč")

    ret = response.choices[0].message.content
    if ret is None:
        return {"sheetData": sheetData, "title": title}

    result = json.loads(ret)

    # Ensure result is a dict
    if not isinstance(result, dict):
        return {"sheetData": sheetData, "title": title}

    return result


def final_smart_ai_fix(draft_song_text: str, cropped_image_data) -> dict:
    """
    Apply final AI-based validation and correction to the full song sheet.

    This function runs TWO SEPARATE AI STEPS:
    1. OCR cleanup + chord validation (with image)
    2. Section structure correction (text-only)

    Args:
        draft_song_text: The draft song sheet text in custom format
        cropped_image_data: Full song page image (BGR format)

    Returns:
        Dictionary with 'title' and 'sheetData' keys
    """
    print("\n🔍 Step 1: OCR cleanup and chord validation...")

    # Save image to temp folder
    current_file = Path(__file__).resolve()
    image_parser_root = current_file.parent.parent.parent.parent
    temp_dir = image_parser_root / "temp" / "final_corrections"
    temp_dir.mkdir(parents=True, exist_ok=True)

    rand_suffix = uuid.uuid4().hex[:8]
    output_path = temp_dir / f"full_song_{rand_suffix}.jpg"
    cv2.imwrite(str(output_path), cropped_image_data)

    try:
        # STEP 1: OCR cleanup and chord validation (with image)
        step1_result = _step1_ocr_cleanup(draft_song_text, str(output_path))
        title = step1_result.get("title", "")
        final_sheet_data = step1_result.get("sheetData", draft_song_text)
        print("✅ Step 1 completed")

        # STEP 2: Section structure correction (text-only)
        print("🔍 Step 2: Section structure correction...")
        step2_result = _step2_section_correction(final_sheet_data, title)
        final_sheet_data = step2_result.get("sheetData", final_sheet_data)
        title = step2_result.get("title", title)
        print("✅ Step 2 completed")

        # Print total cost summary
        from ai import get_price
        price_info = get_price()
        print(f"\n💵 Total AI cost: {price_info['cost_czk_formatted']} ({price_info['total_tokens']} tokens)\n")

        return {
            "title": title,
            "sheetData": final_sheet_data
        }

    except Exception as exc:
        print(f"❌ Final AI correction failed: {exc}\n")
        return {"title": "", "sheetData": draft_song_text}
