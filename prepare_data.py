"""
Этап 1: Подготовка данных
- Загружаем GTSRB
- Делим train на train/val
- Применяем аугментацию
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from utils import IMG_SIZE, NUM_CLASSES, set_seed

DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'Train')
TEST_CSV = os.path.join(DATA_DIR, 'Test.csv')

class GTSRBDataset(Dataset):
    """Кастомный датасет GTSRB"""
    def __init__(self, samples, transform=None):
        self.samples = samples  # список (путь, класс)
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_train_samples():
    """Собираем пути картинок для трейна"""
    samples = []
    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(TRAIN_DIR, str(class_id))
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.ppm')):
                samples.append((os.path.join(class_dir, fname), class_id))
    return samples


def get_test_samples():
    """Тестовый набор из Test.csv"""
    df = pd.read_csv(TEST_CSV)
    samples = []
    for _, row in df.iterrows():
        path = os.path.join(DATA_DIR, row['Path'])
        samples.append((path, int(row['ClassId'])))
    return samples


def get_transforms():
    """Аугментации для трейна и обычные преобразования для теста"""
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tf, test_tf


def get_dataloaders(batch_size=64):
    """Создаём загрузчики данных"""
    set_seed(42)
    train_tf, test_tf = get_transforms()
    
    all_train = get_train_samples()
    print(f'Всего обучающих картинок: {len(all_train)}')
    
    # Делим 80% train / 20% val
    full_dataset = GTSRBDataset(all_train, transform=train_tf)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Для валидации лучше без аугментации
    val_ds.dataset = GTSRBDataset(all_train, transform=test_tf)
    
    test_samples = get_test_samples()
    test_ds = GTSRBDataset(test_samples, transform=test_tf)
    print(f'Тестовых картинок: {len(test_ds)}')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Проверяем что данные загружаются
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches:   {len(val_loader)}')
    print(f'Test batches:  {len(test_loader)}')
    
    # Берём один батч для проверки
    imgs, labels = next(iter(train_loader))
    print(f'Размер батча: {imgs.shape}, метки: {labels[:5].tolist()}')
    print('Данные готовы!')