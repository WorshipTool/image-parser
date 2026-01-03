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
from ai import send_prompt_with_schema


def _step1_ocr_cleanup(draft_song_text: str, image_path: str, debug: bool = False) -> dict:
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

    system_prompt = (
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
        "- if there are chords in square brackets, keep them as they are. DO NOT REMOVE VALID CHORDS\n\n"

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
        "}"
    )

    user_prompt = f"DRAFT SONG SHEET:\n{draft_song_text}"

    result = send_prompt_with_schema(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_schema=schema,
        image_path=image_path,
        schema_name="ocr_cleanup"
    )

    # Ensure result is a dict
    if not isinstance(result, dict):
        return {"title": "", "sheetData": draft_song_text}

    return result


def _step2_section_correction(sheetData: str, title: str) -> dict:
    """
    STEP 2: Section structure correction (text-only, no image).

    Args:
        sheetData: OCR-cleaned sheet from step 1
        title: Song title from step 1

    Returns:
        Dictionary with 'sheetData' and 'title' keys
    """
    schema = {
        "type": "object",
        "properties": {
            "sheetData": {"type": "string"},
            "title": {"type": "string"}
        },
        "required": ["sheetData", "title"],
        "additionalProperties": False
    }

    system_prompt = (
        "You are a strict SONG SHEET NORMALIZER.\n"
        "Input is an already-parsed song in a custom format, but it may contain OCR noise.\n"
        "Your job is to make the sheetData CLEAN, CONSISTENT, and MEANINGFUL.\n"
        "Every character must have a purpose; otherwise remove it.\n\n"

        "SHEET FORMAT RULES:\n"
        "- Inline chords are in square brackets ONLY: [C], [Dm7], [F/A]\n"
        "- Smallcased chords, like [e], [g]..., are okay, it means minor chords.\n"
        "- Section tags are in curly braces at the start of a line.\n\n"

        "VALID CHORD RULES (STRICT):\n"
        "- A chord token must look like a real chord.\n"
        "- Allowed characters inside [] are only: A-G a-g 0-9 # b / +\n"
        "- If a bracketed token contains ANY other character (quotes, commas, weird symbols), fix if obvious.\n"
        "- If it cannot be fixed into a valid chord, REMOVE the whole chord token including brackets.\n"
        "- Never output lyrics inside [].\n"
        "- IMPORTANT! DO NOT REMOVE CHORDS\n"
        "- Chords can be even in section hints, keep them as they are. Only remove the section hint\n"
        "- Never output unclosed/dangling brackets.\n\n"

        "NOISE CLEANUP (STRICT):\n"
        "- Remove stray punctuation/quotes/strokes that are not part of words or chords.\n"
        "- Remove isolated garbage tokens like: \"—\", \"--\", \"=\", random commas/quotes, single symbols.\n"
        "- Keep normal punctuation in lyrics only if it clearly belongs (comma, period).\n\n"

        "HINT REMOVAL (IMPORTANT):\n"
        "- The input may contain structural hints like: 'Ref', 'Refrén', 'Chorus', 'Bridge', '1.', '2.', 'Verse', etc.\n"
        "- If for example Ref is present, it indicates that the section is a chorus. Tag it with right section tag.\n"
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
        "NO SECTION CAN BE EMPTY"

        "TITLE RULES:\n"
        "- The title MUST be returned exactly as given in the input.\n"
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
        "}"
    )

    user_prompt = (
        "Do not remove chords in square brackets in no way, if they are present.\n\n"
        "Your main task is to fix section tags based on hints and clean up any remaining OCR noise.\n\n"
        f"INPUT TITLE:\n{title}\n\n"
        f"INPUT SHEETDATA:\n{sheetData}"
    )

    result = send_prompt_with_schema(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_schema=schema,
        image_path=None,  # No image for step 2
        schema_name="section_correction"
    )

    # Ensure result is a dict
    if not isinstance(result, dict):
        return {"sheetData": sheetData, "title": title}

    return result


def final_smart_ai_fix(draft_song_text: str, cropped_image_data, debug: bool = False) -> dict:
    """
    Apply final AI-based validation and correction to the full song sheet.

    This function runs TWO SEPARATE AI STEPS:
    1. OCR cleanup + chord validation (with image)
    2. Section structure correction (text-only)

    Args:
        draft_song_text: The draft song sheet text in custom format
        cropped_image_data: Full song page image (BGR format)
        debug: If True, save intermediate texts to temp folder

    Returns:
        Dictionary with 'title' and 'sheetData' keys
    """

    # Save image to temp folder
    current_file = Path(__file__).resolve()
    image_parser_root = current_file.parent.parent.parent.parent
    temp_dir = image_parser_root / "temp" / "final_corrections"
    temp_dir.mkdir(parents=True, exist_ok=True)

    rand_suffix = uuid.uuid4().hex[:8]
    output_path = temp_dir / f"full_song_{rand_suffix}.jpg"
    cv2.imwrite(str(output_path), cropped_image_data)

    # Debug: Save text before step 1
    if debug:
        debug_path_before_step1 = temp_dir / f"text_00_before_step1_{rand_suffix}.txt"
        with open(debug_path_before_step1, 'w', encoding='utf-8') as f:
            f.write("=== DRAFT SONG TEXT (BEFORE STEP 1) ===\n\n")
            f.write(draft_song_text)
        print(f"  📝 Saved draft text: {debug_path_before_step1}")

    try:
        # STEP 1: OCR cleanup and chord validation (with image)
        step1_result = _step1_ocr_cleanup(draft_song_text, str(output_path))
        title = step1_result.get("title", "")
        final_sheet_data = step1_result.get("sheetData", draft_song_text)

        # Debug: Save text after step 1 (before step 2)
        if debug:
            debug_path_after_step1 = temp_dir / f"text_01_after_step1_{rand_suffix}.txt"
            with open(debug_path_after_step1, 'w', encoding='utf-8') as f:
                f.write("=== AFTER STEP 1: OCR CLEANUP ===\n\n")
                f.write(f"Title: {title}\n\n")
                f.write(final_sheet_data)
            print(f"  📝 Saved step 1 output: {debug_path_after_step1}")

        # TODO: Second step is not working well, remove chords and so on
        # STEP 2: Section structure correction (text-only)
        # print("🔍 Step 2: Section structure correction...")
        # step2_result = _step2_section_correction(text_after_step1, title)
        # final_sheet_data = step2_result.get("sheetData", text_after_step1)
        # title = step2_result.get("title", title)

        # Debug: Save text after step 2 (final)
        # if debug:
        #     debug_path_after_step2 = temp_dir / f"text_02_after_step2_FINAL_{rand_suffix}.txt"
        #     with open(debug_path_after_step2, 'w', encoding='utf-8') as f:
        #         f.write("=== AFTER STEP 2: SECTION CORRECTION (FINAL) ===\n\n")
        #         f.write(f"Title: {title}\n\n")
        #         f.write(final_sheet_data)
        #     print(f"  📝 Saved final output: {debug_path_after_step2}")

        return {
            "title": title,
            "sheetData": final_sheet_data
        }

    except Exception as exc:
        print(f"❌ Final AI correction failed: {exc}\n")
        return {"title": "", "sheetData": draft_song_text}

    finally:
        # Clean up temporary image if not in debug mode
        if not debug:
            try:
                import os
                if os.path.exists(str(output_path)):
                    os.remove(str(output_path))
            except Exception as e:
                print(f"Warning: Failed to clean up temp file: {e}")
