import os
import glob
import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION & SETUP ---
if os.name == 'nt':  # windows
    # Constants for the pipeline
    DATASET_PATH = r"data\font_hanzi_images"
    FONT_PATH = r"data\font\Source Han Sans CN Light.otf"  # <--- IMPORTANT: Change this to your font file name
    TEST_FONT_PATH = r"data\font\YRDZST Semibold.ttf"
    MODEL_PATH = "classical_pipeline_hanzi_classifier.pkl"
    IMG_SIZE = (64, 64) # Standard size for each character image

    df = pd.read_csv(r"data\characters.csv")
else:   # Mac e linux com / --> endereço correto!
    # Constants for the pipeline
    DATASET_PATH = r"data/font_hanzi_images"
    FONT_PATH = r"data/font/Source Han Sans CN Light.otf"  # <--- IMPORTANT: Change this to your font file name
    TEST_FONT_PATH = r"data/font/YRDZST Semibold.ttf"
    MODEL_PATH = "classical_pipeline_hanzi_classifier.pkl"
    IMG_SIZE = (64, 64) # Standard size for each character image

    df = pd.read_csv(r"data/characters.csv")

    
HANZI_LIST = df["汉字"].tolist()
PINYIN_MAP = dict(zip(df["汉字"], df["拼音"].str.split("|")))

# Parameters for easy tweaking
HOG_PPC = (4, 4) # Pixels per cell
HOG_CPB = (2, 2) # Cells per block
HOG_ORIENTATIONS = 6 # Number of gradient bins
HOG_BLOCK_NORM = 'L2-Hys' # Normalization method

def create_dataset_from_fonts():
    """
    Generates a dataset of Hanzi images from a TTF font file.
    Each character will be saved as an image in its own folder.
    """
    if os.path.exists(DATASET_PATH):
        print(f"Dataset folder '{DATASET_PATH}' already exists. Skipping generation.")
        return

    print(f"Generating dataset in '{DATASET_PATH}'...")
    os.makedirs(DATASET_PATH, exist_ok=True)
    
    # Check if font file exists
    if not os.path.exists(FONT_PATH):
        print(f"ERROR: Font file '{FONT_PATH}' not found!")
        print("Please download a Chinese font (e.g., a .ttf file) and place it in the script's directory.")
        return

    font = ImageFont.truetype(FONT_PATH, size=IMG_SIZE[0] - 10)

    for char in HANZI_LIST:
        char_dir = os.path.join(DATASET_PATH, char)
        os.makedirs(char_dir, exist_ok=True)

        # Create a single image for each character for this example
        image = Image.new('L', IMG_SIZE, color=0)  # 'L' for grayscale
        draw = ImageDraw.Draw(image)

        # Get bounding box of the character to center it
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]

        # Calculate position to center the character
        position = ((IMG_SIZE[0] - char_width) / 2, (IMG_SIZE[1] - char_height) / 2)
        draw.text(position, char, font=font, fill=255)

        image_path = os.path.join(char_dir, f"{char}.png")
        image.save(image_path)
    print("Dataset generation complete.")


# --- 2. FEATURE EXTRACTION & TRAINING ---

def extract_features(visualize=False):
    """
    Loads images from the dataset, resizes them, and extracts HoG features.
    Optionally displays HoG visualization for a few images.
    """
    features = []
    labels = []
    print("Extracting features from dataset...")

    for idx, char_dir in enumerate(glob.glob(os.path.join(DATASET_PATH, "*"))):
        char_label = os.path.basename(char_dir)
        for image_path in glob.glob(os.path.join(char_dir, "*.png")):
            image = np.array(Image.open(image_path).convert("L"))
            resized_image = cv2.resize(image, IMG_SIZE)

            if visualize and idx < 3:
                from skimage import exposure
                feature_vector, hog_image = hog(resized_image,
                                                orientations=HOG_ORIENTATIONS,
                                                pixels_per_cell=HOG_PPC,
                                                cells_per_block=HOG_CPB,
                                                block_norm=HOG_BLOCK_NORM,
                                                visualize=True)
                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                cv2.imshow(f"HOG for {char_label}", hog_image_rescaled.astype('uint8'))
                cv2.waitKey(500)
                cv2.destroyAllWindows()
            else:
                feature_vector = hog(resized_image,
                                     orientations=HOG_ORIENTATIONS,
                                     pixels_per_cell=HOG_PPC,
                                     cells_per_block=HOG_CPB,
                                     block_norm=HOG_BLOCK_NORM)
            features.append(feature_vector)
            labels.append(char_label)
            
    print(f"Extracted {len(features)} feature vectors.")
    return np.array(features), np.array(labels)

def train_hanzi_classifier(features, labels):
    print("Training the classifier...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(labels)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(set(labels)), yticklabels=sorted(set(labels)))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    joblib.dump(model, MODEL_PATH)
    print(f"Classifier trained and saved to '{MODEL_PATH}'.")


# --- 3. PREDICTION PIPELINE ---

def preprocess_and_segment(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return [], None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray)  
    cv2.waitKey(500)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow("Binary (Inverted)", binary)  
    cv2.waitKey(500)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    rois = []
    for (x, y, w, h) in bounding_boxes:
        if w > 10 and h > 10:
            roi = binary[y:y+h, x:x+w]
            cv2.imshow(f"ROI ({x},{y})", roi)  
            cv2.waitKey(300)
            rois.append(((x, y, w, h), roi))

    cv2.destroyAllWindows()
    return rois, image


def predict_on_image(image_path, save_result=True):
    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found. Please train the model first.")
        return

    model = joblib.load(MODEL_PATH)
    rois, original_image = preprocess_and_segment(image_path)
    if original_image is None: return

    print(f"Found {len(rois)} potential characters.")

    for (box, roi) in rois:
        (x, y, w, h) = box
        resized_roi = cv2.resize(roi, IMG_SIZE)

        feature_vector = hog(resized_roi,
                             orientations=HOG_ORIENTATIONS,
                             pixels_per_cell=HOG_PPC,
                             cells_per_block=HOG_CPB,
                             block_norm=HOG_BLOCK_NORM)

        prediction = model.predict([feature_vector])[0]
        pinyin = PINYIN_MAP.get(prediction, "?")

        print(f"Predicted: '{prediction}' ({pinyin}) at x={x}, y={y}, w={w}, h={h}")  

        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype(FONT_PATH, 15)
        except IOError:
            font = ImageFont.load_default()
        draw.text((x, y - 20), f"{prediction} ({pinyin})", font=font, fill=(0, 255, 0))
        original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    cv2.imshow("Prediction Result", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_result:
        result_path = "annotated_prediction.png"
        cv2.imwrite(result_path, original_image)
        print(f"Annotated result saved as '{result_path}'")


def create_test_image():
    """Creates a simple test image with some Hanzis for prediction."""
    text_to_draw = "你好世界"
    test_image_path = "test_image.png"
    
    font = ImageFont.truetype(TEST_FONT_PATH, size=50)
    bbox = ImageDraw.Draw(Image.new('L', (1,1))).textbbox((0,0), text_to_draw, font=font)
    img_width = bbox[2] - bbox[0] + 40
    img_height = bbox[3] - bbox[1] + 40
    
    image = Image.new('L', (img_width, img_height), color=0)
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), text_to_draw, font=font, fill=255)
    image.save(test_image_path)
    return test_image_path


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    create_dataset_from_fonts()
    features, labels = extract_features(visualize=False)
    if features.size > 0:
        train_hanzi_classifier(features, labels)
    else:
        print("No features extracted. Cannot train model. Please check your dataset.")

    print("\nCreating a test image for prediction...")
    test_image_file = create_test_image()
    print("Running prediction on the test image...")
    predict_on_image(test_image_file)
