import re
import math
from dataclasses import dataclass
from statistics import median
from typing import List
import cv2
import uuid
from pathlib import Path

from .sheet import Sheet
from ..ocr.read_word_data import ReadWordData
from .line import Line
from .section import Section
from .smart_line_correction import smart_lines_correction
from ai import send_image_and_question

def insert_str(string, str_to_insert, index):
    return string[:index] + str_to_insert + string[index:]

def is_it_chord(chord):
    chord_pattern = r'\b[A-Ga-gHh][#b]?[mM]?[0-9]?(/[A-Ga-gHh][#b]?)?\b'

    matches = re.fullmatch(chord_pattern, chord)
    if matches:
        return True
    else:
        return False

def get_title_from_sections(sections: list[Section]) -> tuple[str, list[Section]]:
    if len(sections) == 0 or len(sections[0].lines) == 0:
        return "Název písně", sections
    
    section = sections[0]
    if len(section.lines) == 1 and section.lines[0].chordLinePossibility <= 0.5: 
        return str(section.lines[0]), sections[1:]

    for section in sections:
        for line in section.lines:
            if line.chordLinePossibility <= 0.5:
                return str(line), sections

    return "Název písně", sections
    # title = ""
    # for index, line in enumerate(lines):
    #     isLyrics = line[1]<0.5
    #     if(isLyrics):
    #         words = line[2]
    #         for wordIndex, word in enumerate(words):
    #             if(wordIndex>0):
    #                 previousEndX = words[wordIndex-1]['bounds']['left'] + words[wordIndex-1]['bounds']['width']
    #                 currentStartX = word['bounds']['left']
    #                 spaceSize = abs(currentStartX - previousEndX)
                    
    #                 width = word['bounds']['width']
    #                 sizeThreshold = width * 2
    #                 if(spaceSize > sizeThreshold):
    #                     break
    #             title += word['text'] + " "
    #         if(index==0): del lines[0]
    #         break
    # return title.strip(), lines

def split_lines_to_sections(lines: list[Line]) -> list[Section]:

    # Calculate average Y line distance
    minLineDistance = 0
    maxLineDistance = 0
    avgLineDistance = 0
    lineDistances : list[float] = []

    for index, line in enumerate(lines):
        if(index==0): continue

        distance = abs(line.centerY - lines[index-1].centerY)
        lineDistances.append(distance)

        if(minLineDistance>distance or index==1): 
            minLineDistance = distance
        if(maxLineDistance<distance or index==1): 
            maxLineDistance = distance
        avgLineDistance += distance

    avgLineDistance /= len(lines)

    # Split lines to sections by distance threshold
    distanceThreshold = avgLineDistance*1.5
    sections : list[Section] = []
    for index, line in enumerate(lines):
        if(index==0): isNewSection = True
        else: 
            distance = lineDistances[index-1]
            isNewSection = distance > distanceThreshold

        if(isNewSection):
            sections.append(Section([line]))
        else:
            sections[len(sections)-1].lines.append(line)
    return sections

def filter_lines(lines, show=False):
    for index, line in enumerate(lines):
        words = line[2]
        if(len(words)>1): continue;
        if(line[1]>0.5): continue;
        
        firstWord = words[0]
        if(len(firstWord['text'])>1): continue;
        
        if(show): print("Deleting:",firstWord['text'])
        del lines[index]
        
        
    return lines


def lines_to_formatted_string(lines: list[Line]) -> str:
    data = ""

    for lineIndex, line in enumerate(lines):
            isChord = line.chordLinePossibility>0.5
            isAboveLyrics = lineIndex<len(lines)-1 and lines[lineIndex+1].chordLinePossibility<0.5 and len(lines[lineIndex+1].words) > 0
            isBelowChords = lineIndex>0 and lines[lineIndex-1].chordLinePossibility>0.5

            lineData = ""
            if isChord:
                if isAboveLyrics:
    #                 Add chords to the lyrics string, on correct places
    #                 Parse text from lyrics
                    lyricsLine = lines[lineIndex+1]
                    lyricsText = ""
                    for word in lyricsLine.words:
                        lyricsText+=word.text + " "
                    lyricsText.strip()

                    editedLyrics = lyricsText

    #                 Get left and right text X positions
                    left = lyricsLine.words[0].bounds.left
                    right = lyricsLine.words[len(lyricsLine.words)-1].bounds.left + lyricsLine.words[len(lyricsLine.words)-1].bounds.width

                    for chord in line.words[::-1]:
                        chordX = chord.bounds.left
                        text = chord.text
                        
                        if(right-left == 0): continue
                            
                        placeInLyricsCoef = (chordX - left) / (right-left)
                        charIndex = math.floor(placeInLyricsCoef*len(lyricsText))
                        charIndex = max(charIndex, 0)
                        charIndex = min(charIndex, len(lyricsText)-1)

                        chordText = "["+text+"]"
                        editedLyrics = insert_str(editedLyrics, chordText, charIndex)



                    lineData+=editedLyrics
                else:
                    for word in line.words:
                        lineData+="["+word.text+"]"

            else:
                
                if not isBelowChords:
                    for word in line.words:
                        lineData+=word.text + " "
            if not lineData == "":
                data += lineData
                data += "\n"

    
    data = data[:-1]
    return data

def sections_to_formatted_string(sections: list[Section]) -> str:
    sectionCountPerName : dict[str, int]= {}

    sectionStrings = []
    for sectionIndex,section in enumerate(sections):

        if not section.name in sectionCountPerName:
            sectionCountPerName[section.name] = 0
        sectionCountPerName[section.name] += 1

        sectionString = "{"+section.name+str(sectionCountPerName[section.name])+"}"

        linesString = lines_to_formatted_string(section.lines)
        
        if not linesString == "":
            sectionString += linesString
       
        sectionStrings.append(sectionString)
        
        
    return  "\n\n".join(sectionStrings)


# předpoklad: ReadWordData má .text a .bounds (left, top, width, height)

def read_word_list_to_lines(wordData: List[ReadWordData]) -> List[Line]:
    if not wordData:
        return []

    # 1) seřadit shora dolů (a lehce zleva, ať je to deterministické)
    words = sorted(
        wordData,
        key=lambda w: (w.bounds.top + w.bounds.height / 2, w.bounds.left)
    )

    # 2) robustní "typická výška" (mix akordů+textu)
    heights = [w.bounds.height for w in words]
    h_med = median(heights)
    # tolerance: cca půl řádku; upravitelné
    y_tol = max(6.0, 0.55 * h_med)

    lines: List[Line] = []

    def word_center_y(w) -> float:
        return w.bounds.top + w.bounds.height / 2

    def word_center_x(w) -> float:
        return w.bounds.left + w.bounds.width / 2

    for w in words:
        cy = word_center_y(w)
        chordPoss = float(is_it_chord(w.text))

        # 3) najdi nejbližší řádek podle Y (ne první co projde)
        best_i = -1
        best_dist = 1e9
        for i, line in enumerate(lines):
            dist = abs(line.centerY - cy)
            if dist < best_dist:
                best_dist = dist
                best_i = i

        if best_i == -1 or best_dist > y_tol:
            # nový řádek
            lines.append(Line(cy, chordPoss, [w]))
        else:
            line = lines[best_i]

            # 4) update centerY (inkrementální průměr)
            n = len(line.words)
            line.centerY = (line.centerY * n + cy) / (n + 1)

            # update chord likelihood průměrem
            line.chordLinePossibility = (line.chordLinePossibility * n + chordPoss) / (n + 1)

            line.words.append(w)

    # 5) seřadit slova v řádcích zleva doprava
    for line in lines:
        line.words.sort(key=lambda w: w.bounds.left)

    # 6) a ještě seřadit řádky podle Y
    lines.sort(key=lambda l: l.centerY)

    return lines
def get_title(titleData: list[ReadWordData]) -> str:
    lines = read_word_list_to_lines(titleData)
    sections = split_lines_to_sections(lines)
    title, sections = get_title_from_sections(sections)
    return title

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

        "SONG SHEET FORMAT:\n"
        "- Lyrics contain inline chords in square brackets: [C], [Dm7], [F/A]\n"
        "- Sections are marked with tags like {1S}, {2S}, {1R}, {I}, {B}, {O}\n\n"

        "YOUR TASKS:\n\n"

        "1) EXTRACT SONG TITLE\n"
        "- Read the song title from the IMAGE.\n"
        "- Return it EXACTLY as written (preserve diacritics and casing).\n\n"

        "2) FIX OCR ERRORS IN LYRICS\n"
        "- Correct typos, broken words, wrong letters.\n"
        "- Fix diacritics and casing errors.\n"
        "- Remove random symbols that don't belong to lyrics.\n"
        "- Do NOT paraphrase or change meaning.\n\n"

        "3) FIX CHORD SYMBOLS\n"
        "- Valid chord characters: A–G, a–g, 0–9, #, b, /, +\n"
        "- Fix obvious chord typos (e.g. Dmi7 → Dm7, Emi → Em).\n"
        "- REMOVE chords with invalid characters.\n"
        "- Do NOT add new chords.\n\n"

        "4) PRESERVE STRUCTURE\n"
        "- Keep ALL section tags EXACTLY AS THEY ARE.\n"
        "- Do NOT rename, merge, split, or remove section tags.\n"
        "- Keep line order and formatting unchanged.\n"
        "- Only fix the TEXT and CHORDS.\n\n"

        "CONSTRAINTS:\n"
        "- Do NOT invent lyrics or chords.\n"
        "- Do NOT analyze song structure.\n"
        "- Do NOT reorder lines.\n"
        "- When unsure, keep the original.\n\n"

        "RETURN ONLY VALID JSON:\n"
        "{\n"
        "  \"title\": \"<exact song title from image>\",\n"
        "  \"sheetData\": \"<corrected text with original section tags>\"\n"
        "}\n\n"

        "DRAFT SONG SHEET:\n"
        f"{draft_song_text}"
    )

    result = send_image_and_question(image_path, prompt, json_schema=schema)

    # Ensure result is a dict
    if not isinstance(result, dict):
        return {"title": "", "sheetData": draft_song_text}

    return result


def _step2_section_correction(cleaned_sheet_text: str) -> dict:
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
            "sheetData": {"type": "string"}
        },
        "required": ["sheetData"],
        "additionalProperties": False
    }

    prompt = (
        "You are correcting SECTION TAGS in a song sheet.\n"
        "The lyrics and chords are already correct. Focus ONLY on section structure.\n\n"

        "SONG SHEET FORMAT:\n"
        "- Lyrics contain inline chords: [C], [Dm7], [F/A]\n"
        "- Sections are marked with tags:\n"
        "  {1S}, {2S}, {3S}, ... = verses (stanzas)\n"
        "  {1R}, {2R}, ... = choruses (refrains)\n"
        "  {I} = Intro, {O} = Outro\n"
        "  {M} = Interlude / instrumental (chord-only block between sections)\n\n"

        "CRITICAL TAG PLACEMENT RULE:\n"
        "- A section tag may appear ONLY on the FIRST line of that section.\n"
        "- All following lines in the same section MUST NOT start with a section tag.\n\n"

        "CHORUS RULE (IMPORTANT):\n"
        "- A chorus section MUST contain at least ONE lyric/text line.\n"
        "- If a block contains ONLY chords (no normal words), it is NOT a chorus.\n"
        "- Chord-only blocks between sections must be tagged as {M}, not {R}.\n\n"

        "YOUR TASKS:\n\n"

        "1) DETECT CHORUSES\n"
        "- Find lyrical blocks that REPEAT (same or very similar lyrics).\n"
        "- Mark them as {1R}, {2R}, ...\n"
        "- Apply the tag ONLY on the first line of the block.\n\n"

        "2) RENAME VERSES\n"
        "- Rename non-chorus lyric sections as {1S}, {2S}, {3S}, ... in order.\n"
        "- Apply the tag ONLY on the first line of each verse.\n\n"

        "3) DETECT INTRO / OUTRO / INTERLUDES\n"
        "- Chord-only section at the START → {I}\n"
        "- Chord-only section at the END → {O}\n"
        "- Chord-only section BETWEEN verses/choruses → {M}\n"
        "- Apply the tag ONLY on the first line of the block.\n\n"

        "4) FIX SECTION BOUNDARIES\n"
        "- Merge sections only if they clearly belong together.\n"
        "- Split sections only if they are clearly different parts.\n"
        "- Remove section tags only if the section is empty or invalid.\n\n"

        "CONSTRAINTS:\n"
        "- Do NOT change lyrics, chords, or line text.\n"
        "- Do NOT reorder lines.\n"
        "- ONLY change section tags and their placement.\n"
        "- Preserve the original flow of the song.\n\n"

        "RETURN ONLY VALID JSON:\n"
        "{\n"
        "  \"sheetData\": \"<same content with corrected section tags>\"\n"
        "}\n\n"

        "SONG SHEET:\n"
        f"{cleaned_sheet_text}"
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
        return {"sheetData": cleaned_sheet_text}

    result = json.loads(ret)

    # Ensure result is a dict
    if not isinstance(result, dict):
        return {"sheetData": cleaned_sheet_text}

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
        cleaned_text = step1_result.get("sheetData", draft_song_text)
        print("✅ Step 1 completed")

        # STEP 2: Section structure correction (text-only)
        print("🔍 Step 2: Section structure correction...")
        step2_result = _step2_section_correction(cleaned_text)
        final_sheet_data = step2_result.get("sheetData", cleaned_text)
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

def format(dataData:list[ReadWordData], inputImagePath: str, cropedImageData) -> Sheet:

    title = get_title(dataData)


    lines = read_word_list_to_lines(dataData)

    # Apply smart corrections
    lines = smart_lines_correction(lines, cropedImageData)

    # Print lines
    for line in lines:
        print(f"[{line.avgConfidence, line.chordLinePossibility}] {[word.text for word in line.words]}")

    sections = split_lines_to_sections(lines)
    data = sections_to_formatted_string(sections)

    # Apply final AI-based validation and correction
    corrected_result = final_smart_ai_fix(data, cropedImageData)
    if corrected_result.get("title"):
        title = corrected_result["title"]
    if corrected_result.get("sheetData"):
        data = corrected_result["sheetData"]

    return Sheet(title, data, inputImagePath, cropedImageData)
