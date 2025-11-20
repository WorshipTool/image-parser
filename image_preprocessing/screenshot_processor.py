import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple


class ScreenshotProcessor:
    """
    Zpracovává screenshoty - používá YOLO model pro detekci oblasti s písní
    a provádí crop na relevantní oblast.
    """

    def __init__(self, yolo_model_path: str):
        """
        Inicializace procesoru pro screenshoty.

        Args:
            yolo_model_path: Cesta k YOLO modelu
        """
        self.model = YOLO(yolo_model_path)

    def process(self, image_path: str, image: np.ndarray) -> np.ndarray:
        """
        Zpracuje screenshot - detekuje oblast s písní a provede crop.

        Args:
            image_path: Cesta k obrázku (pro YOLO model)
            image: Načtený obrázek

        Returns:
            Oříznutý obrázek s oblastí písně
        """
        # Spustíme YOLO detekci
        results = self.model.predict(image_path, conf=0.25, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            # Pokud se nic nedetekuje, vrátíme originální obrázek
            return image

        # Najdeme největší detekovanou oblast (pravděpodobně "sheet")
        boxes = results[0].boxes
        best_box = None
        max_area = 0

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                best_box = (int(x1), int(y1), int(x2), int(y2))

        # Pokud jsme našli validní box, cropneme
        if best_box is not None:
            x1, y1, x2, y2 = best_box

            # Přidáme malý padding (5% z každé strany)
            h, w = image.shape[:2]
            padding_x = int((x2 - x1) * 0.05)
            padding_y = int((y2 - y1) * 0.05)

            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(w, x2 + padding_x)
            y2 = min(h, y2 + padding_y)

            # Crop
            cropped = image[y1:y2, x1:x2]
            return cropped

        return image

    def detect_all_sheets(self, image_path: str) -> list:
        """
        Detekuje všechny písně v obrázku.

        Args:
            image_path: Cesta k obrázku

        Returns:
            List detekovaných oblastí [(x1, y1, x2, y2, confidence, class_name), ...]
        """
        results = self.model.predict(image_path, conf=0.25, verbose=False)

        detections = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = results[0].names[cls]

                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'class': class_name
                })

        return detections
