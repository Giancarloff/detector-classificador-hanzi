import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import pandas as pd

# ======== CONFIGURAÇÕES ========
BATCH_SIZE = 64
NUM_EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= CNN CLASSIFIER ==========
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# ======== TRANSFORMAÇÕES (escala de cinza) ========
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ======== DADOS ========
class HanziDataset(Dataset):
    def __init__(self, image_dir, hanzi_list, transform=None):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform
        self.hanzi2idx = {hanzi: idx for idx, hanzi in enumerate(hanzi_list)}
        self.valid_paths = []
        for img_path in self.image_paths:
            hanzi = os.path.basename(img_path)[0]
            if hanzi in self.hanzi2idx:
                self.valid_paths.append(img_path)
        self.image_paths = self.valid_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 'L' para grayscale
        hanzi = os.path.basename(img_path)[0]
        label = self.hanzi2idx[hanzi]
        if self.transform:
            image = self.transform(image)
        return image, label

# ======== CARREGAR DADOS DE TREINO E TESTE ========
hanzi_list = pd.read_csv("data/characters.csv")["汉字"].tolist()
hanzi_list = hanzi_list[:10]  # Use only the first 10 characters for quick testing
NUM_CLASSES = len(hanzi_list)

train_dataset = HanziDataset("data/images/QIJIC Regular", hanzi_list, transform=transform)
test_dataset = HanziDataset("data/images/shijuef.com(gongfanmianfeiti) Regular", hanzi_list, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======== INICIALIZAR MODELO, CRITÉRIO E OTIMIZADOR ========
model = SimpleCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======== TREINAMENTO ========
print("Iniciando o treinamento...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

# ======== AVALIAÇÃO ========
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