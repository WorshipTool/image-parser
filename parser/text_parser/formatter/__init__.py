import re
import math
from dataclasses import dataclass
from statistics import median
from typing import List

from .sheet import Sheet
from ..ocr.read_word_data import ReadWordData
from .line import Line
from .section import Section
from .smart_line_correction import smart_lines_correction
from .utils import save_lines_visualization
from .final_smart_correction import final_smart_ai_fix

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

def format(dataData:list[ReadWordData], inputImagePath: str, cropedImageData, debug: bool = False) -> Sheet:

    title = get_title(dataData)


    lines = read_word_list_to_lines(dataData)

    # Apply smart corrections
    # lines = smart_lines_correction(lines, cropedImageData, debug=debug)

    # Save visualization of corrected lines with raw OCR data
    if debug:
        save_lines_visualization(cropedImageData, lines, dataData)

    # Print lines
    # for line in lines:
    #     print(f"[{line.avgConfidence, line.chordLinePossibility}] {[word.text for word in line.words]}")

    sections = split_lines_to_sections(lines)
    data = sections_to_formatted_string(sections)

    # Apply final AI-based validation and correction
    corrected_result = final_smart_ai_fix(data, cropedImageData, debug=debug)
    if corrected_result.get("title"):
        title = corrected_result["title"]
    if corrected_result.get("sheetData"):
        data = corrected_result["sheetData"]

    return Sheet(title, data, inputImagePath, cropedImageData)
