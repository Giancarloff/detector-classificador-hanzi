"""Algumas fontes não possuem todos os caracteres, 
então é necessário limpar as imagens geradas para 
remover caracteres que não estão presentes na fonte."""
import os
from PIL import Image
import numpy as np

def delete_images(image_paths):
    """Deleta as imagens cujos caminhos estão na lista image_paths."""
    for img_path in image_paths:
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Deletado: {img_path}")
        else:
            print(f"Arquivo não encontrado: {img_path}")

def is_mostly_white(image_path, threshold=0.98, center_only=False,imagem_direta=False):
    """
    Retorna True se a imagem for quase toda branca.
    threshold: porcentagem mínima de pixels brancos (0~1).
    center_only: se True, verifica apenas a região central.
    """
    if imagem_direta:
        arr = np.array(image_path)
    else:
        if not os.path.exists(image_path):
            print(f"Arquivo não encontrado: {image_path}")
            return False
        img = Image.open(image_path).convert("L")
        arr = np.array(img)
    if center_only:
        h, w = arr.shape
        ch, cw = h // 4, w // 4
        arr = arr[ch:3*ch, cw:3*cw]  # pega o centro
    white = np.sum(arr > 250)
    total = arr.size
    return (white / total) >= threshold

if __name__ == "__main__":
    # Caminho para a pasta de imagens
    image_folder = "data/images"

    total_images = 0
    total_white_images = 0

    for sub_folder in os.listdir(image_folder):
        # Se não for uma pasta, pula
        if not os.path.isdir(os.path.join(image_folder, sub_folder)):
            continue

        sub_folder_path = os.path.join(image_folder, sub_folder)
    
        # Lista de imagens a serem verificadas
        image_paths = [os.path.join(sub_folder_path, img) for img in os.listdir(sub_folder_path) if img.endswith('.png')]
        
        # Filtrar imagens que são quase todas brancas
        white_images = [img for img in image_paths if is_mostly_white(img, center_only=True)]
        
        # Deletar as imagens filtradas
        delete_images(white_images)

        # Soma as imagens totais
        total_images += len(image_paths) - len(white_images)
        total_white_images += len(white_images)

        i = 0
        for img in image_paths:
            i += 1
        print(f"{sub_folder}: {i} imagens restantes.")
    
    print("Processamento concluído.")
    print(f"Total de imagens deletadas: {total_white_images}")
    print(f"Total de imagens restantes: {total_images}")

    
