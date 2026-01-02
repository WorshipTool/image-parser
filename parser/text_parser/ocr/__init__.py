import pytesseract
from pytesseract import Output

from .read_format_converter import ReadFormatConverter

def read(image):
    result = pytesseract.image_to_data(image, output_type=Output.DICT,  config='-l ces+slk --psm 11 ');
    formatted = ReadFormatConverter.convert_to_custom_format(result)

    #Filter out words with low confidence
    formatted = [word for word in formatted if word.confidence >= 45]

    return formatted