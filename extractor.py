import cv2
import numpy as np
import os
import sys

# --- CONFIGURATION ---
IMG_SIZE = (64, 64)
MIN_CHAR_SIZE = 15  # Minimum width/height of contour to be considered valid
DEBUG_VISUALIZE = True  # Set to False to disable debug visualizations
OUTPUT_DIR = "segmented_characters"

# Remove overlapping boxes using Non-Maximum Suppression
def non_max_suppression_fast(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]

    areas = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(y2)

    pick = []
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[idxs[:-1]]

        idxs = np.delete(
            idxs, np.concatenate(([len(idxs) - 1], np.where(overlap > overlapThresh)[0]))
        )
    return boxes[pick].astype("int")


def extract_hanzi_from_image(image_path, save=True):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize (invert for white text on black)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Dilation to merge nearby strokes ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find contours from dilated image (merged strokes)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = [b for b in bounding_boxes if b[2] > MIN_CHAR_SIZE and b[3] > MIN_CHAR_SIZE]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rois = []

    for idx, (x, y, w, h) in enumerate(bounding_boxes):
        roi = binary[y:y+h, x:x+w]  # Use original binary for cleaner chars
        roi_resized = cv2.resize(roi, IMG_SIZE)
        rois.append(roi_resized)

        if save:
            out_path = os.path.join(OUTPUT_DIR, f"char_{idx}.png")
            cv2.imwrite(out_path, roi_resized)

        if DEBUG_VISUALIZE:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #cv2.imshow(f"Char {idx}", roi_resized)
            cv2.waitKey(300)

    if DEBUG_VISUALIZE:
        cv2.imshow("Merged Bounding Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Extracted {len(rois)} character regions (after merging).")
    return rois

def extract_hanzi_with_mser(image_path, save=True):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply contrast enhancement and denoising first
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Adaptive thresholding followed by dilation (like the earlier version)
    binary = cv2.adaptiveThreshold(gray, 255,
                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV,
                                25, 15)

    # Morphological dilation to connect strokes of the same Hanzi
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Now run MSER on the *dilated image* (converted back to grayscale if needed)
    mser_input = dilated.copy()
    mser = cv2.MSER.create(delta=5, min_area=80, max_area=8000)
    regions, _ = mser.detectRegions(mser_input)

    # Get bounding boxes from MSER regions
    bounding_boxes = []
    for p in regions:
        x, y, w, h = cv2.boundingRect(p)
        if w > MIN_CHAR_SIZE and h > MIN_CHAR_SIZE:
            bounding_boxes.append((x, y, w, h))

    bounding_boxes = non_max_suppression_fast(bounding_boxes)
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rois = []

    for idx, (x, y, w, h) in enumerate(bounding_boxes):
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, IMG_SIZE)
        rois.append(roi)

        if save:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"char_{idx}.png"), roi)

        if DEBUG_VISUALIZE:
            #cv2.imshow(f"Char {idx}", roi)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.waitKey(200)

    if DEBUG_VISUALIZE:
        cv2.imshow("Detected MSER Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Extracted {len(rois)} character regions using MSER.")
    return rois

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Uso: diretorio/imagem -opção")
        print("Opções: -d para contorno, -m para MSER")
    image = sys.argv[1]
    option = sys.argv[2]
    if os.path.exists(image):
        if option == "-d":
            extract_hanzi_from_image(image)
        elif option == "-m":
            extract_hanzi_with_mser(image)
        else:
            print(f"Opção não reconhecida: {option}")

    else:
        print("Image not found.")
