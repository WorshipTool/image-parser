# Image-Parser

## Popis

Image-Parser je Python program pro automatickou extrakci křesťanských písní z fotografií. Program detekuje písně na obrázku, rozpozná text a převede ho do strukturovaného JSON formátu pro webovou aplikaci Chvalotce.cz.

## Instalace

### Local Development

```bash
git clone https://github.com/WorshipTool/image-parser.git && cd image-parser
pip install -r requirements.txt
python prepare.py
```

**Note:** The requirements include the WorshipTool Bridge module for service discovery, which is installed directly from GitHub. This is optional - the server will run without it if not needed.

### Docker (Recommended for Production)

```bash
# Setup
cp .env.example .env
# Edit .env and add OPENAI_API_KEY (optional, for AI corrections)

# Build and start
docker compose build
docker compose up -d

# Access server at http://localhost:6610
# Swagger docs at http://localhost:6610/docs
# RQ Dashboard at http://localhost:6610/board

# View logs
docker compose logs -f

# Stop services
docker compose down
```

## Použití

### CLI (Command Line Interface)

```bash
# Basic usage - parse single image
python main.py image.jpg

# Multiple images
python main.py img1.jpg img2.jpg img3.jpg

# With AI corrections (slower but more accurate)
python main.py image.jpg --ai

# With debug output
python main.py image.jpg --debug

# Custom output directory
python main.py *.jpg -o output/

# All options
python main.py image.jpg --ai --debug -o results/
```

**Podporované formáty:** JPG, PNG

### HTTP Server API (Local Development)

```bash
# Requires Redis running locally
# Start server
python -m server

# Start worker (separate terminal)
python -m server.worker
```

Server documentation: See `server/README.md`

## Project Structure

```
image-parser/
├── main.py                    # CLI entry point
├── parser/                    # Parser module (CLI tool)
│   ├── parse.py              # Main parser implementation
│   ├── get_sheet_components.py
│   ├── text_parser/          # OCR and text formatting
│   ├── paper_detection/      # Paper detection and transformation
│   ├── paper_transform/      # Perspective correction
│   └── sheet_detection/      # YOLO-based sheet detection
├── server/                    # HTTP API server
│   ├── __main__.py           # Module entry point
│   ├── app.py                # Flask application
│   ├── api.py                # Parser API wrapper
│   └── README.md             # Server documentation
├── ai/                        # AI correction modules
└── temp/                      # Temporary files (auto-created)
```

## Jak to funguje

### 1. Předzpracování obrazu

-   **Detekce perspektivy:** Oprava rotace a perspektivního zkreslení pomocí `PhotoPerspectiveFixer`
-   Normalizace orientace dokumentu pro lepší detekci

### 2. Detekce objektů (YOLO model)

Program používá vlastní natrénovaný **YOLOv8 model** (`yolo8best.pt`), který detekuje tři typy objektů:

-   **`sheet`** - celá stránka písně (obsahuje vše)
-   **`title`** - titulek písně
-   **`data`** - tělo písně (text, akordy, sloka/refrén)

Model je natrénován na fotografiích křesťanských zpěvníků a proložek s písněmi. Dokáže identifikovat umístění jednotlivých částí písně i v komplikovanějších layoutech nebo při šikmém naskenování.

### 3. Seskupení detekovaných oblastí

-   Filtrování duplicit (odstraní menší detekce uvnitř větších)
-   Seskupení title + data do logických celků (pomocí `SongDetectGroup`)
-   Párování titulků s obsahem podle prostorové blízkosti

### 4. OCR - rozpoznání textu

-   Pro každou detekovanou oblast se provádí OCR (optical character recognition)
-   Získání textu včetně souřadnic jednotlivých slov (`ReadWordData`)

### 5. Formátování výstupu

-   Strukturování rozpoznaného textu do JSON formátu
-   Rozpoznání struktury: sloky, refrénu, akordů

## Contributions

Contributions are welcome! Open an issue or create a pull request.
