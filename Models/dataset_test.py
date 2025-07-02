from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import pandas as pd
import os
import random
import json
"""
This script generates images of Hanzi characters using various fonts,
applies distortions to the images, and saves them in a specified directory.
It also compresses the generated images into zip files for each font.
NOTE: for adding variants to dataset run with "distorced_only"
NOTE: IA generativa usada para funções de distorcão (integralmente)
e geração (parcialmente).
"""
print(os.getcwd())

#==================== Configurations ========================
par = "data"
parent_font = par + "/font"

with open('data/font_hanzi_dict.json', "r", encoding="utf-8") as f:
    font_hanzi_dict = json.load(f)
print(font_hanzi_dict.keys())

#"""
FONT_PATH_LIST = [f"{par}/Source Han Sans CN Light.otf"]

#"""
#FONT_PATH_LIST = [f"{parent_font}/QIJIC Regular.ttf"]
  # Path to your font file
FONT_SIZE = 64              # Size of the characters
IMAGE_SIZE = (84, 84)       # Image dimensions (width, height)
BACKGROUND_COLOR = 255      # Background color (branco)
TEXT_COLOR = "black"        # Text color
OUTPUT_DIR = '/home/nm/Imagens/test'
#OUTPUT_DIR = par + "/DATASET/validation/"
#OUTPUT_DIR = par + "/DATASET/test/"
hanzi_list = pd.read_csv('data/characters.csv')["汉字"].tolist()
def distort_image(image):
    # Rotação aleatória
    image = image.rotate(random.uniform(-10, 10))
    # Blur
    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))
    # Brilho
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))
    return image

def generate_images_by_character(font_paths, hanzi_list, font_hanzi_dict_path, distorced_only=False, distorced_num=1, parent_output=OUTPUT_DIR):
    # Carrega o dicionário de suporte de caracteres
    with open(font_hanzi_dict_path, "r", encoding="utf-8") as f:
        font_hanzi_dict = json.load(f)

    # Cria diretório raíz se não existir
    os.makedirs(parent_output, exist_ok=True)

    # Prepara contadores para progresso
    total_chars = len(hanzi_list)
    processed_chars = 0

    for hanzi in hanzi_list:
        char_dir = os.path.join(parent_output, hanzi)
        os.makedirs(char_dir, exist_ok=True)

        # Contador para verificar se algum font gerou este caractere
        generated_count = 0

        for font_path in font_paths:
            font_name = os.path.splitext(os.path.basename(font_path))[0]

            # Verifica se a fonte suporta o caractere
            if hanzi not in font_hanzi_dict.get(font_name, []):
                continue

            try:
                # Cria a imagem base
                font = ImageFont.truetype(font_path, FONT_SIZE)
                image = Image.new("L", IMAGE_SIZE, BACKGROUND_COLOR)
                draw = ImageDraw.Draw(image)

                # Centraliza o caractere
                bbox = font.getbbox(hanzi)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                position = ((IMAGE_SIZE[0] - w) // 2, (IMAGE_SIZE[1] - h) // 2)
                draw.text(position, hanzi, font=font, fill=TEXT_COLOR)

                # Salva imagem limpa
                if not distorced_only:
                    clean_path = os.path.join(char_dir, f"{font_name}_clean.png")
                    image.save(clean_path)

                # Gera imagens distorcidas
                for i in range(distorced_num):
                    distorted = distort_image(image.copy())  # Usar cópia para não afetar original
                    distorted_path = os.path.join(char_dir, f"{font_name}_dist_{i}.png")
                    distorted.save(distorted_path)

                generated_count += 1

            except Exception as e:
                print(f"Erro ao processar {font_name} para {hanzi}: {str(e)}")

        # Atualiza progresso
        processed_chars += 1
        percent = (processed_chars / total_chars) * 100
        if generated_count > 0:
            status = f"({generated_count} fonts)"
        else:
            status = "(no supported fonts)"

        if processed_chars % 100 == 0 or processed_chars == total_chars:
            print(f"Progresso: {percent:.2f}% | Caractere: {hanzi} {status}")

generate_images_by_character(
    font_paths=FONT_PATH_LIST,
    hanzi_list=hanzi_list,
    font_hanzi_dict_path='data/font_hanzi_dict.json',
    distorced_num=1,          # versões distorcidas por fonte
    parent_output=OUTPUT_DIR,
    distorced_only=False
)
print(f"Imagens geradas em {OUTPUT_DIR}")