# Quick Start Guide

Jednoduchý průvodce pro image preprocessing.

## Použití

### CLI (příkazová řádka)

```bash
# Zpracovat obrázek (vytvoří temp/image_processed.png)
./preprocess.sh -i obrazek.jpg

# Zpracovat a uložit do konkrétního souboru
./preprocess.sh -i obrazek.jpg -o output.png
```

**To je vše!** Žádné další parametry, vždy stejné zpracování.

**Poznámka:** Výchozí výstup jde do `temp/` složky, která se automaticky vytvoří.

## Co se děje při zpracování?

1. **Automatická detekce typu** (screenshot/photo/scan)
2. **Screenshot**: YOLO crop na oblast s písní
3. **Photo**: perspektivní transformace a narovnání
4. **Grayscale**: převod na odstíny šedi
5. **Denoising**: odstranění šumu
6. **Bez threshold**: zachová odstíny šedi (ne binární)

## Python API

```python
from image_preprocessing import ImagePreprocessor

# Inicializace
preprocessor = ImagePreprocessor("yolo8best.pt")

# Zpracování (vrátí cestu k výstupu v temp/)
output_path = preprocessor.preprocess("image.jpg")
print(f"Výsledek: {output_path}")  # temp/image_processed.png

# S vlastním výstupem
output_path = preprocessor.preprocess("image.jpg", "output.png")
```

## Troubleshooting

### "No module named 'dill'"
Používáš systémový Python místo venv!

**Řešení**: Použij wrapper skript:
```bash
./preprocess.sh -i obrazek.jpg
```

### "YOLO model not found"
Model `yolo8best.pt` musí být ve stejné složce.

### "Image Not Found"
Zkontroluj cestu k obrázku.

## Výstup

- **Formát**: Grayscale PNG (odstíny šedi)
- **Velikost**: Cca 300-400 KB
- **Obsah**: Grayscale s denoisingem, bez threshold
- **Optimalizováno**: Pro OCR (Tesseract)
