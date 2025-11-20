# Image-Parser

## Popis

Image-Parser je Python program pro automatickou extrakci křesťanských písní z fotografií. Program detekuje písně na obrázku, rozpozná text a převede ho do strukturovaného JSON formátu pro webovou aplikaci Chvalotce.cz.

## Instalace

```bash
git clone https://github.com/WorshipTool/image-parser.git && cd image-parser
pip install -r requirements.txt
python prepare.py
```

## Použití

```bash
python main.py -o output.json -i cesta_k_obrazku.jpg
python main.py -o output.json -i obrazek1.jpg obrazek2.png
```

**Podporované formáty:** JPG, PNG

**Poznámka:** AI analýza je vypnutá. Bude se používat samostatně pro částečný processing.

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
