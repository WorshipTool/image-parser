import pytesseract
from pytesseract import Output

from .read_format_converter import ReadFormatConverter

def read(image):
    result = pytesseract.image_to_data(image, output_type=Output.DICT,  config='-l ces+slk --psm 6 ');
    formatted = ReadFormatConverter.convert_to_custom_format(result)
    return formatted