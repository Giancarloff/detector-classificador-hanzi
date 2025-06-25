import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import pandas as pd

# ======== CONFIGURAÇÕES ========
BATCH_SIZE = 64
NUM_EPOCHS = 55
LR = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = 15

# ========= HEAD NN ==========
class CLASSIFIER(nn.Module):
    def __init__(self, in_f, out_f):
        super(CLASSIFIER, self).__init__()
        self.fc1 = nn.Linear(in_f, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128,out_f)
 

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x =self.fc3(x)

        return x

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

        match image_dir:
            case list():
                self.image_paths = []
                for dir in image_dir:
                    self.image_paths.extend(os.path.join(dir, f) 
                                        for f in os.listdir(dir) if f.endswith('.png'))
            case _:
                print(type(image_dir))
                self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        
        self.transform = transform
        self.hanzi2idx = {hanzi: idx for idx, hanzi in enumerate(hanzi_list)}
        self.valid_paths = []
        for img_path in self.image_paths:
            hanzi = os.path.basename(img_path)[0]
            if hanzi in self.hanzi2idx: # Verifica se o caractere está na lista
                self.valid_paths.append(img_path)
        self.image_paths = self.valid_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        hanzi = os.path.basename(img_path)[0]
        label = self.hanzi2idx[hanzi]
        if self.transform:
            image = self.transform(image)
        return image, label

# ======== CARREGAR DADOS DE TREINO E TESTE ========
hanzi_list = pd.read_csv("data/characters.csv")["汉字"].tolist()
hanzi_list = hanzi_list[:CLASSES]  # Use only the first 10 characters for quick testing
NUM_CLASSES = len(hanzi_list)

parent_data = "/home/nm/Imagens/images"

train_dataset = HanziDataset([parent_data + "/YRDZST Semibold",parent_data + "/sucaijishikufangti Regular",parent_data + "/Source Han Sans CN Light",
                  parent_data + "/ShouShuti Regular",parent_data + "/shijuef.com(gongfanmianfeiti) Regular",parent_data + "/QIJIC Regular"], hanzi_list, transform=transform)
test_dataset = HanziDataset([parent_data + "/HanyiSentyPagoda Regular",parent_data + "/AZPPT_1_1436212_19 Regular"], hanzi_list, transform=transform)
#test_dataset = HanziDataset("data/images/QIJIC Regular", hanzi_list, transform=transform)  # teste sobre o próprio dataset
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======== CARREGAR CONVNEXT BASE PRÉ-TREINADO ========
resnet = models.resnet50(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

in_features = resnet.fc.in_features
resnet.fc = CLASSIFIER(in_features, NUM_CLASSES)

for param in resnet.fc.parameters():
    param.requires_grad = True

model = resnet.to(DEVICE)

# ======== FUNÇÃO DE CUSTO, OTIMIZADOR E SCHEDULER ========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR, weight_decay=1e-5) # Adicionado weight_decay (ver ponto 4)

# Reduz a LR por um fator de 0.1 a cada 20 épocas
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

if __name__ == "__main__":
    # ======== TREINAMENTO ========
    print("Iniciando o treinamento...")
    best_loss = np.inf
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

        
        # Atualiza a taxa de aprendizagem
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Modelo salvo em 'best_model.pth' com loss: {best_loss:.4f}")

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # ======== AVALIAÇÃO RÁPIDA ========
    print("\nIniciando a avaliação...")
    model.eval()
    correct, total, b_total, b_cor = 0, 0, 0 , 0
    best_model = resnet.to(DEVICE)
    best_model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            best_out = best_model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, b_predicted = torch.max(best_out.data, 1)
            total += labels.size(0) # patch size
            b_total += labels.size(0)
            correct += (predicted == labels).sum().item()
            b_cor += (b_predicted == labels).sum().item()
    acc = 100 * correct / total
    best_acc = 100 * b_cor / total

    print(f"Accuracy: {acc:.2f}%")
    print(f"Previous accuracy: {best_acc:.2f}%")

    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model_acc.pth')
        print(f"Modelo salvo em 'best_model_acc.pth' com loss: {best_acc:.2f} %")

