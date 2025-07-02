from fastai.vision.all import *
import os

MODEL_PATH = 'Models/EfficientNET/EfficientnetB0-hanzi_0(1).pkl'

learn = load_learner(MODEL_PATH)

def predict_image(img_path):
    pred_class, pred_idx, probs = learn.predict(img_path)
    print(f"Predicted class: {pred_class}")
    print(f"Probability: {probs[pred_idx]:.4f}")
    return pred_class, probs[pred_idx].item()

if __name__ == "__main__":
    
    if len(sys.argv) != 1:
        print("Uso: diretorio/imagem")
    image = sys.argv[1]
    if os.path.exists(image):
        predict_image(image)
    else:
        print("Image not found.")