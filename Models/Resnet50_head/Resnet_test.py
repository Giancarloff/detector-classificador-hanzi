import torch
import pandas as pd
from torchvision import models
from Resnet_train import HanziDataset, CLASSIFIER
from torchvision import transforms, models
from torch.utils.data import DataLoader

# ========= CONFIG ==========
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======== TRANSFORMAÇÕES (estilo ImageNet) ========
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # padrão ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

# ======== CARREGAR DADOS DE TREINO E TESTE ========
hanzi_list = pd.read_csv("data/characters.csv")["汉字"].tolist()
hanzi_list = hanzi_list[:15]  # Use only the first 10 characters for quick testing
NUM_CLASSES = len(hanzi_list)

#test_dataset = HanziDataset(["data/images/HanyiSentyPagoda Regular","data/images/AZPPT_1_1436212_19 Regular"], hanzi_list, transform=transform)
test_dataset = HanziDataset(["/home/nm/Imagens/test/HanyiSentyPagoda Regular","/home/nm/Imagens/test/AZPPT_1_1436212_19 Regular","/home/nm/Imagens/test/YRDZST Semibold","/home/nm/Imagens/test/sucaijishikufangti Regular","/home/nm/Imagens/test/Source Han Sans CN Light"],
                             hanzi_list, transform=transform)  # teste sobre o próprio dataset

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Parâmetros iguais aos do treino
resnet = models.resnet50(pretrained=False)
in_features = resnet.fc.in_features
resnet.fc = CLASSIFIER(in_features, NUM_CLASSES)

# Carregar pesos
#resnet.load_state_dict(torch.load('Models/Resnet50_head/best_model.pth', map_location=DEVICE))
resnet.load_state_dict(torch.load('Models/Resnet50_head/fine_tuned_model.pth', map_location=DEVICE))

# Colocar em modo de avaliação
print("\nIniciando a avaliação...")
resnet.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

import matplotlib.pyplot as plt

# Plot de setores para corretos e incorretos
incorrect = total - correct
labels = ['Corretos', 'Incorretos']
sizes = [correct, incorrect]
colors = ['#4CAF50', '#F44336']

plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Proporção de Previsões Corretas e Incorretas')
plt.axis('equal')
plt.show()

print(f"Accuracy: {100 * correct / total:.2f}%")