from .read_word_data import ReadWordData, Bounds

class ReadFormatConverter:

    def getWordDataWithIndex(index, data):
        return ReadWordData(
            data["text"][index],
            Bounds(
                data["left"][index],
                data["top"][index],
                data["width"][index],
                data["height"][index]
            ),
            data["conf"][index]
        
        )

    def convert_to_custom_format(input):
        words : list[ReadWordData] = []
        for i in range(len(input["text"])):
            wordData = ReadFormatConverter.getWordDataWithIndex(i, input)

            if wordData.text == "":
                continue

            words.append(wordData)

        return words