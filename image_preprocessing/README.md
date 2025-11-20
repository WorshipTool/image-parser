# Image Preprocessing Module

Modul pro předzpracování obrázků písní před OCR procesem.

## Přehled

Tento modul implementuje první krok OCR workflow podle `ocr_gpt_workflow.txt`:
1. Načtení obrázku
2. Detekce typu obrázku (screenshot/photo/scan)
3. Specifické předzpracování podle typu
4. Základní předzpracování (grayscale, denoising, threshold)

## Struktura modulu

```
image_preprocessing/
├── __init__.py                 # Export hlavní třídy
├── preprocessor.py             # Hlavní třída ImagePreprocessor
├── screenshot_processor.py     # Zpracování screenshotů (YOLO detekce)
├── photo_processor.py          # Zpracování fotek (perspektivní transformace)
└── utils.py                    # Pomocné funkce
```

## Použití

### CLI (příkazová řádka)

```bash
# Zpracovat obrázek (vytvoří temp/image_processed.png)
./preprocess.sh -i image.jpg

# Zpracovat a uložit do konkrétního souboru
./preprocess.sh -i image.jpg -o output.png
```

**Parametry:**
- `-i, --input` - Vstupní obrázek (povinný)
- `-o, --output` - Výstupní soubor (volitelný, výchozí: `temp/input_processed.png`)
- `--model` - Cesta k YOLO modelu (volitelný, výchozí: `yolo8best.pt`)

**Zpracování (vždy stejné):**
1. Automatická detekce typu (screenshot/photo/scan)
2. Screenshot: YOLO crop
3. Photo: perspektivní transformace
4. Grayscale + denoising (bez threshold)

### Python API

```python
from image_preprocessing import ImagePreprocessor

# Inicializace
preprocessor = ImagePreprocessor("yolo8best.pt")

# Zpracování (vrátí cestu k výstupu)
output_path = preprocessor.preprocess("image.jpg")
print(f"Výsledek: {output_path}")

# S vlastním výstupem
output_path = preprocessor.preprocess("image.jpg", "output.png")
```

### Typy obrázků

#### Screenshot
- Používá YOLO model pro detekci oblasti s písní
- Provádí crop na relevantní oblast
- Automaticky detekován podle cesty obsahující "screenshot"

```python
from image_preprocessing.utils import ImageType

processed = preprocessor.preprocess("screenshot.png", ImageType.SCREENSHOT)

# Detekce všech písní v obrázku
detections = preprocessor.screenshot_processor.detect_all_sheets("screenshot.png")
for det in detections:
    print(f"{det['class']}: {det['confidence']:.2f}")
```

#### Photo
- Detekuje papír v obrázku
- Provádí perspektivní transformaci (narovnání)
- Automaticky detekuje a opravuje rotaci
- Automaticky detekován podle cesty obsahující "photo"

```python
processed = preprocessor.preprocess("photo.jpg", ImageType.PHOTO)
```

#### Scan
- Pouze základní předzpracování
- Žádná speciální detekce nebo transformace
- Automaticky detekován podle cesty obsahující "scan"

```python
processed = preprocessor.preprocess("scan.jpg", ImageType.SCAN)
```

## Předzpracování

### Screenshot Processing
1. YOLO detekce oblasti s písní (class: sheet, title, data)
2. Crop na největší detekovanou oblast + 5% padding
3. Základní předzpracování

### Photo Processing
1. Detekce papíru pomocí edge detection
2. Nalezení 4 rohů papíru
3. Perspektivní transformace (four-point transform)
4. Detekce a korekce rotace textu
5. Základní předzpracování

### Základní předzpracování (pro všechny typy)
1. **Grayscale**: Převod na černobílý obrázek
2. **Denoising**: Odstranění šumu (fastNlMeansDenoising)
3. **Threshold**: Otsu's binarization pro optimální práh

## API

### `ImagePreprocessor`

#### `__init__(yolo_model_path: str)`
Inicializace preprocessoru s cestou k YOLO modelu.

#### `preprocess(image_path: str, image_type: Optional[ImageType] = None) -> np.ndarray`
Předzpracuje obrázek podle jeho typu. Vrací grayscale binární obrázek.

**Parametry:**
- `image_path`: Cesta k obrázku
- `image_type`: Typ obrázku (nebo None pro automatickou detekci)

**Vrací:** numpy array (grayscale, uint8)

#### `preprocess_keep_color(image_path: str, image_type: Optional[ImageType] = None) -> np.ndarray`
Předzpracuje obrázek, ale zachová barevnou verzi (bez grayscale a threshold).

**Vrací:** numpy array (BGR, uint8)

### Pomocné funkce (`utils.py`)

#### `detect_image_type(image: np.ndarray, image_path: str = None) -> ImageType`
Automaticky detekuje typ obrázku.

#### `order_points(pts: np.ndarray) -> np.ndarray`
Seřadí 4 body čtyřúhelníku (TL, TR, BR, BL).

#### `four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray`
Provede perspektivní transformaci.

#### `rotate_image(image: np.ndarray, angle: float) -> np.ndarray`
Otočí obrázek o daný úhel.

#### `detect_rotation_angle(image: np.ndarray) -> float`
Detekuje úhel rotace textu v obrázku.

## Testy

Spuštění testů:
```bash
venv/bin/python -m pytest tests/test_image_preprocessing.py -v
```

Demo skript:
```bash
venv/bin/python demo_preprocessing.py
```

Výsledky demo skriptu se ukládají do `tmp/preprocessing_results/`.

## Příklady výsledků

Po spuštění `demo_preprocessing.py`:
- `screenshot_N_color.png` - Screenshot s cropem na oblast písně (barevný)
- `screenshot_N_gray.png` - Screenshot po plném předzpracování (grayscale + threshold)
- `photo_N_color.png` - Fotka po perspektivní transformaci (barevná)
- `photo_N_gray.png` - Fotka po plném předzpracování (grayscale + threshold)

## Požadavky

- OpenCV (`cv2`)
- NumPy
- Ultralytics YOLO
- YOLO model (`yolo8best.pt`)

Všechny požadavky jsou specifikovány v `requirements.txt`.

## Další kroky

Po předzpracování obrázků následuje:
1. Tesseract OCR (viz `image_reader/`)
2. Složení slov do řádků
3. Rozlišení akordových a textových řádků
4. GPT analýza (viz `ocr_gpt_workflow.txt`)
