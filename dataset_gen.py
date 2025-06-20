from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import pandas as pd
import zipfile
import os
import random
from concurrent.futures import ProcessPoolExecutor
from image_cleaner import is_mostly_white
import time

""" 
This script generates images of Hanzi characters using various fonts, 
applies distortions to the images, and saves them in a specified directory. 
It also compresses the generated images into zip files for each font.
"""

# Configurations
#FONT_PATH_LIST = ["font/data/YRDZST Semibold.ttf","font/data/sucaijishikufangti Regular.ttf","font/data/Source Han Sans CN Light.otf",
#                  "font/data/ShouShuti Regular.ttf","font/data/shijuef.com(gongfanmianfeiti) Regular.ttf","font/data/QIJIC Regular.ttf",
#                  "font/data/HanyiSentyPagoda Regular.ttf","font/data/AZPPT_1_1436212_19 Regular.ttf","font/data/__________ Regular.ttf",
#                  "font/data/____ Regular.otf"]
FONT_PATH_LIST = ["data/font/QIJIC Regular.ttf"]
ZIP_PATH_LIST = [f"data/images/{font.strip(" ").replace("data/", "").replace(".ttf", "")}.zip" for font in FONT_PATH_LIST]
  # Path to your font file
FONT_SIZE = 64              # Size of the characters
IMAGE_SIZE = (84, 84)       # Image dimensions (width, height)
BACKGROUND_COLOR = "white"  # Background color
TEXT_COLOR = "black"        # Text color
OUTPUT_DIR = [f"data/images/{font.strip(" ").replace("data/", "").replace(".ttf", "")}" for font in FONT_PATH_LIST] # Folder to save images


def distort_image(image):
    # Rotação aleatória
    image = image.rotate(random.uniform(-10, 10))
    # Blur
    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))
    # Brilho
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))
    return image

def generate_images_for_font(font_path, hanzi_list, hanzi_num,distorced_only=False,distorced_num=1):
    font_name = os.path.splitext(os.path.basename(font_path))[0]
    output_dir = f"data/images/{font_name}"
    zip_path = f"{output_dir}.zip"
    os.makedirs(output_dir, exist_ok=True)
    font = ImageFont.truetype(font_path, FONT_SIZE)

    for i, hanzi in enumerate(hanzi_list, 1):
        image = Image.new("L", IMAGE_SIZE, BACKGROUND_COLOR)
        if not distorced_only:
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), hanzi, font=font, fill=TEXT_COLOR)
            image.save(os.path.join(output_dir, f"{hanzi}.png"))

        # Imagem distorcida
        for n_d in range(distorced_num):  
            distorted = distort_image(image)
            img_path_aug = os.path.join(output_dir, f"{hanzi}{n_d}_aug.png")
            # Verifica se a imagem distorcida é quase toda branca
            if is_mostly_white(distorted, center_only=True, imagem_direta=True):
                continue
            distorted.save(img_path_aug)

        percent = (i / hanzi_num) * 100
        if i % 100 == 0 or i == hanzi_num:
            print(f"{font_name}: {percent:.2f}%")

    print(f"\nGenerated {hanzi_num} images in '{output_dir}'.")
    """
    # Compactar imagens
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img_name in os.listdir(output_dir):
            img_path = os.path.join(output_dir, img_name)
            arcname = os.path.relpath(img_path, output_dir)
            zipf.write(img_path, arcname)
    print(f"Compressed images into '{zip_path}'.")
    """

if __name__ == "__main__":
    init = time.perf_counter()
    # Carregar CSV e configs
    df = pd.read_csv("data/characters.csv")
    hanzi_list = df["汉字"].tolist()
    hanzi_num = len(hanzi_list)

    # Paralelizar por fonte
    with ProcessPoolExecutor() as executor:
        futures = []
        for font_path in FONT_PATH_LIST:
            futures.append(executor.submit(generate_images_for_font, font_path, hanzi_list, hanzi_num,distorced_only=True,distorced_num=3))
        for future in futures:
            future.result()  # Espera todas terminarem
    elapsed = time.perf_counter() - init
    print(f"\nAll images generated and compressed in {elapsed:.2f} seconds.")