"""
Этап 2: Обучение 5 разных архитектур
1. SimpleCNN  - своя простая сверточная сеть
2. ResNet18   - предобученная
3. MobileNetV2 - предобученная (легкая)
4. EfficientNet-B0 - предобученная
5. VGG11      - предобученная
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from utils import DEVICE, NUM_CLASSES, set_seed, save_json, count_parameters
from prepare_data import get_dataloaders  # переименуй 1_prepare_data.py в prepare_data.py

MODELS_DIR = 'models'
RESULTS_DIR = 'results'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============ Архитектура 1: своя простая CNN ============
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))


def get_model(name):
    """Возвращает модель по имени"""
    if name == 'SimpleCNN':
        return SimpleCNN(NUM_CLASSES)
    
    elif name == 'ResNet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        return m
    
    elif name == 'MobileNetV2':
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
        return m
    
    elif name == 'EfficientNetB0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
        return m
    
    elif name == 'VGG11':
        m = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, NUM_CLASSES)
        return m
    
    else:
        raise ValueError(f'Неизвестная модель: {name}')


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc='Train', leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        _, pred = out.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Val', leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            _, pred = out.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_model(name, train_loader, val_loader, epochs=5, lr=1e-3):
    """Обучаем одну модель"""
    print(f'\n{"="*60}\nОбучение: {name}\n{"="*60}')
    set_seed(42)
    
    model = get_model(name).to(DEVICE)
    n_params = count_parameters(model)
    print(f'Параметров: {n_params:,}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    
    start = time.time()
    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} | '
              f'train_loss={tr_loss:.4f} acc={tr_acc:.4f} | '
              f'val_loss={val_loss:.4f} acc={val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'{name}.pt'))
    
    train_time = time.time() - start
    
    info = {
        'name': name,
        'params': n_params,
        'epochs': epochs,
        'lr': lr,
        'best_val_acc': best_val_acc,
        'train_time_sec': train_time,
        'history': history
    }
    save_json(info, os.path.join(RESULTS_DIR, f'{name}_train.json'))
    print(f'Лучшая val_acc: {best_val_acc:.4f} | время: {train_time:.1f}с')
    return info


if __name__ == '__main__':
    print(f'Устройство: {DEVICE}')
    train_loader, val_loader, _ = get_dataloaders(batch_size=64)
    
    models_to_train = ['SimpleCNN', 'ResNet18', 'MobileNetV2', 'EfficientNetB0', 'VGG11']
    
    all_results = {}
    for name in models_to_train:
        try:
            result = train_model(name, train_loader, val_loader, epochs=5, lr=1e-3)
            all_results[name] = result
        except Exception as e:
            print(f'Ошибка при обучении {name}: {e}')
    
    save_json(all_results, os.path.join(RESULTS_DIR, 'all_training.json'))
    print('\nВсе модели обучены!')