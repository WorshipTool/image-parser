# Image-Parser

## Description
Image-Parser is a Python program designed as a feature for the web application Chvalotce.cz, serving as a database of Christian songs. All songs are stored in a specific format for proper subsequent display. The purpose of this Python program is to identify a song in a photograph, read it, and convert it to the specified format. Its primary functionality lies in the ability to extract songs from images.

## Installation
To install Image-Parser, use the following steps:

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/WorshipTool/image-parser.git && cd image-parser
   ```

2. Run the preparation script to download necessary files and install dependencies:
    ```bash
    python prepare.py && pip install -r requirements.txt
    ```


## Usage
To run the program, enter the following command in the command line, with the following parameters:

```bash
python main.py -o output_file.json -i path_to_image1 path_to_image2 ...
```

### Parameters:
- `-o output_file.json`: Path to the file where the resulting data will be saved (e.g., `output.json`).
- `-i path_to_image1 path_to_image2 ...`: Paths to the input images to be processed.

### Image Formats
The Image-Parser program supports the following image formats:

- **JPEG (JPG):** Joint Photographic Experts Group format.
- **PNG:** Portable Network Graphics format.

Ensure that your input images are in either JPG or PNG format for optimal results.


## Contributions and Commitments
If you would like to contribute to this project, please open an issue or create a pull request. We welcome improvements!




