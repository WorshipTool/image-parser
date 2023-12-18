from common.bounds import Bounds


class CustomDetect:
    def __init__(self, box, result):
        self.label : str = result.names[int(box.cls)]
        self.confidence : float = box.conf.item()
        self.bounds = Bounds(box.xywh[0][0], box.xywh[0][1], box.xywh[0][2], box.xywh[0][3])
        
        # Crop image
        originalImg = result.orig_img
        self.image = originalImg[int(box.xyxy[0][1]):int(box.xyxy[0][3]), int(box.xyxy[0][0]):int(box.xyxy[0][2])]
        