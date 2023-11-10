import cv2 as cv
import numpy as np

class PhotoPerspectiveFixer:
    
        
    # def fix(image):
    #     global document_contour

    #     WIDTH, HEIGHT = 1920, 1080
    #     document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    #     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #     blur = cv.GaussianBlur(gray, (5, 5), 0)
    #     _, threshold = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    #     contours, _ = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    #     contours = sorted(contours, key=cv.contourArea, reverse=True)

    #     max_area = 0
    #     for contour in contours:
    #         area = cv.contourArea(contour)
    #         if area > 1000:
    #             peri = cv.arcLength(contour, True)
    #             approx = cv.approxPolyDP(contour, 0.015 * peri, True)
    #             if area > max_area and len(approx) == 4:
    #                 document_contour = approx
    #                 max_area = area

    #     cv.drawContours(image, [document_contour], -1, (0, 255, 0), 3)

    #     return image
    

    def fix(image):
        return image # IGNORE PERSPECTIVE FIXING
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        hranice = cv.Canny(gray, 50, 150, apertureSize=3)

        lines = cv.HoughLines(hranice, 1, np.pi / 180, 100)
        uhly = []
        for line in lines:
            for rho, theta in line:
                uhly.append(theta)

        prumer_uhlu = np.mean(uhly)
        anglesInDegrees = np.degrees(prumer_uhlu)

        rotatedImage = image
        if anglesInDegrees != 0:
            rotacni_matice = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), anglesInDegrees, 1)
            rotatedImage = cv.warpAffine(image, rotacni_matice, (image.shape[1], image.shape[0]))

        return rotatedImage
        