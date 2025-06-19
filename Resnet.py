import torch
import torch.nn as nn
from torchvision import models
from util import training

num_classes = 8105  # Número de caracteres Hanzi

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Treinamento com CrossEntropyLoss e otimizador (Adam/SGD)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

