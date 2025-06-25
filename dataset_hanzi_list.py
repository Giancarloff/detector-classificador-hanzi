""" A lot of fonts don't have all hanzis in the character.csv
    this script goes trough all images folder and list the 
    existing hanzis. 
    DO IT BEFORE CREATING THE DISTORTIONS!!!"""
import os
import json

def list_hanzi_per_font(dataset_dir):
    font_hanzi_dict = {}
    for font_folder in os.listdir(dataset_dir):  # font folder is the right already the right name
        folder_path = os.path.join(dataset_dir, font_folder)
        if not os.path.isdir(folder_path):
            continue
        hanzi_set = set()
        for img in os.listdir(folder_path):
            if img.endswith('.png'):
                hanzi = os.path.splitext(img)[0][0]  # Pega o primeiro caractere do nome do arquivo
                hanzi_set.add(hanzi)
        font_hanzi_dict[font_folder] = sorted(list(hanzi_set))
    return font_hanzi_dict

if __name__ == "__main__":
    dataset = "/home/nm/Imagens/images"
    font_hanzi = list_hanzi_per_font(dataset)
    for font, hanzis in font_hanzi.items():
        print(f"{font}: {hanzis}")

        with open("data/font_hanzi_dict.json", "w", encoding="utf-8") as f:
            json.dump(font_hanzi, f, ensure_ascii=False, indent=2)

