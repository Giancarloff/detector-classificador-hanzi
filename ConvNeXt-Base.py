import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import timm
from PIL import Image
import os
import pandas as pd

# ======== CONFIGURAÇÕES ========
BATCH_SIZE = 32
# NUM_CLASSES = 8105  
NUM_EPOCHS = 5
LR = 1e-2
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

# ======== DADOS ========
class HanziDataset(Dataset):
    def __init__(self, image_dir, hanzi_list, transform=None):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform
        # Cria um dicionário: caractere -> índice
        self.hanzi2idx = {hanzi: idx for idx, hanzi in enumerate(hanzi_list)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        # Extrai o primeiro caractere do nome do arquivo
        hanzi = os.path.basename(img_path)[0]
        label = self.hanzi2idx[hanzi]
        if self.transform:
            image = self.transform(image)
        return image, label

# ======== CARREGAR DADOS DE TREINO E TESTE ========
hanzi_list = pd.read_csv("data/characters.csv")["汉字"].tolist()
# <--- CORREÇÃO: Define NUM_CLASSES a partir do tamanho real da lista de caracteres
NUM_CLASSES = len(hanzi_list) 

train_dataset = HanziDataset("data/images/QIJIC Regular", hanzi_list, transform=transform)
test_dataset = HanziDataset("data/images/shijuef.com(gongfanmianfeiti) Regular", hanzi_list, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ======== CARREGAR CONVNEXT BASE PRÉ-TREINADO ========
model = timm.create_model("convnext_base", pretrained=True)

# Congelar todos os pesos (exceto a cabeça de classificação)
for param in model.parameters():
    param.requires_grad = False

# <--- CORREÇÃO: Substituir apenas a camada linear (fc) dentro da cabeça
# 1. Obter o número de features de entrada da camada linear existente
in_features = model.head.fc.in_features
# 2. Substituir a camada linear por uma nova com o número correto de classes
model.head.fc = nn.Linear(in_features, NUM_CLASSES)

# Garantir que os gradientes da nova camada sejam calculados
for param in model.head.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# ======== FUNÇÃO DE CUSTO E OTIMIZADOR ========
criterion = nn.CrossEntropyLoss()
# <--- CORREÇÃO: Passar para o otimizador os parâmetros da nova camada que você quer treinar
optimizer = torch.optim.Adam(model.head.fc.parameters(), lr=LR)

# ======== TREINAMENTO ========
print("Iniciando o treinamento...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

# ======== AVALIAÇÃO RÁPIDA ========
print("\nIniciando a avaliação...")
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
