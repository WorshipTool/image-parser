import json
import sys

def use_arguments():
    return "-default" not in sys.argv


def write_json_to_file(data, file_path):
    jsonString = json.dumps(data, ensure_ascii=False)
    with open(file_path, 'w') as outfile:
        outfile.write(jsonString)

def load_argument (key: str, allKeys: list[str], returnMultipleValues: bool = False):
    args = sys.argv

    if key not in args:
        print("Key" + key + " is missing in arguments")
        sys.exit()

    values : list[str] = []

    keyIndex = args.index(key)
    for index, arg in enumerate(args):
        if index <= keyIndex: continue
        if arg in allKeys: break
        values.append(arg)

    if not returnMultipleValues:
        if len(values) > 0:
            return values[0]
        else:
            return None

    if len(values) == 0:
        return None
    return values
    