from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import zipfile
import os

# Configurations
FONT_PATH = "data/ShouShuti Regular.ttf"  # Path to your font file
FONT_SIZE = 64              # Size of the characters
IMAGE_SIZE = (84, 84)     # Image dimensions (width, height)
BACKGROUND_COLOR = "white"  # Background color
TEXT_COLOR = "black"        # Text color
OUTPUT_DIR = "data/images/test_hanzi_images3" # Folder to save images

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the CSV file
df = pd.read_csv("data/characters.csv")
hanzi_list = df["汉字"].tolist()  # Adjust column name if needed


# Load the font
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

# Generate images
i = 0
for hanzi in hanzi_list:
    i += 1
    # Create a blank image
    image = Image.new("L", IMAGE_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    
    # Draw the character
    draw.text((10, 10), hanzi, font=font, fill=TEXT_COLOR)
    
    # Save the image (filename format: {hanzi}.png)
    image.save(os.path.join(OUTPUT_DIR, f"{hanzi}.png"))

print(f"Generated {i} out of {len(hanzi_list)} images in '{OUTPUT_DIR}'.")

# Compress images into a zip file
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for img_path in image_paths:
        arcname = os.path.relpath(img_path, OUTPUT_DIR)
        zipf.write(img_path, arcname)
print(f"Compressed images into '{ZIP_PATH}'.")