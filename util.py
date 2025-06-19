import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import random

def distort_image(image):
    # Rotação aleatória
    image = image.rotate(random.uniform(-10, 10))
    # Blur
    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))
    # Brilho
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))
    return image

# Treinamento 
def training(model, n_epochs_= 5, lr=0.001, train_loader=None, val_loader=None, crit = nn.CrossEntropyLoss()):
    
    criterion = crit
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = n_epochs_
    train_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    return train_losses, val_accuracies

def plot(losses, accuracy):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(losses, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Validation Accuracy', color=color)  
    ax2.plot(accuracy, color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title('Training Loss and Validation Accuracy')
    plt.show()