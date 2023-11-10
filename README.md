# Image-Parser

## Description
Image-Parser is a Python program designed as a feature for the web application Chvalotce.cz, serving as a database of Christian songs. All songs are stored in a specific format for proper subsequent display. The purpose of this Python program is to identify a song in a photograph, read it, and convert it to the specified format. Its primary functionality lies in the ability to extract songs from images.

## Usage
To run the program, enter the following command in the command line, with the following parameters:

```bash
python main.py -m path_to_model.pth -t path_to_directory -o output_file.json -i path_to_image1 path_to_image2 ...
```

### Parameters:
- `-m path_to_model.pth`: Path to the PyTorch model for song recognition.
- `-t path_to_directory`: Path to the directory where temporary images will be stored.
- `-o output_file.json`: Path to the file where the resulting data will be saved (e.g., `output.json`).
- `-i path_to_image1 path_to_image2 ...`: Paths to the input images to be processed.

## Contributions and Commitments
If you would like to contribute to this project, please open an issue or create a pull request. We welcome improvements!




