"""
Этап 3: Оценка моделей на тесте
- accuracy, precision, recall, F1
- confusion matrix
- скорость инференса
- размер модели
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from utils import DEVICE, NUM_CLASSES, CLASS_NAMES, save_json
from prepare_data import get_dataloaders
from train_models import get_model

MODELS_DIR = 'models'
RESULTS_DIR = 'results'


def predict_all(model, loader):
    """Получаем все предсказания и истинные метки"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            probs = torch.softmax(out, dim=1)
            _, pred = out.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def measure_inference_time(model, loader, n_batches=20):
    """Среднее время обработки одного изображения"""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= n_batches:
                break
            imgs = imgs.to(DEVICE)
            start = time.time()
            _ = model(imgs)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed / imgs.size(0))
    return float(np.mean(times) * 1000)  # в мс


def plot_confusion(cm, name):
    """Confusion matrix"""
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Предсказание')
    plt.ylabel('Истина')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'cm_{name}.png'), dpi=100)
    plt.close()


def evaluate_model(name, test_loader):
    """Оцениваем одну модель"""
    print(f'\n--- Оценка {name} ---')
    model = get_model(name).to(DEVICE)
    weights_path = os.path.join(MODELS_DIR, f'{name}.pt')
    if not os.path.exists(weights_path):
        print(f'Веса не найдены: {weights_path}')
        return None
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    
    preds, labels, probs = predict_all(model, test_loader)
    
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted', zero_division=0)
    rec = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    
    cm = confusion_matrix(labels, preds)
    plot_confusion(cm, name)
    
    inf_time = measure_inference_time(model, test_loader)
    model_size_mb = os.path.getsize(weights_path) / 1024 / 1024
    
    # Топ-5 пар часто путаемых классов
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    confused_pairs = []
    for _ in range(5):
        idx = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
        if cm_no_diag[idx] == 0:
            break
        confused_pairs.append({
            'true': CLASS_NAMES[idx[0]],
            'pred': CLASS_NAMES[idx[1]],
            'count': int(cm_no_diag[idx])
        })
        cm_no_diag[idx] = 0
    
    result = {
        'model': name,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'inference_ms_per_image': inf_time,
        'model_size_mb': model_size_mb,
        'top_confused_pairs': confused_pairs
    }
    
    print(f'Accuracy: {acc:.4f} | F1: {f1:.4f} | '
          f'Inference: {inf_time:.2f}ms | Size: {model_size_mb:.1f}MB')
    
    save_json(result, os.path.join(RESULTS_DIR, f'{name}_eval.json'))
    return result


def make_comparison_table(results):
    """Сводная таблица для отчёта"""
    print('\n' + '='*90)
    print(f'{"Модель":<18}{"Accuracy":<12}{"Precision":<12}{"Recall":<10}{"F1":<10}{"Inf,мс":<10}{"Size,MB":<10}')
    print('='*90)
    for r in results:
        if r is None: continue
        print(f'{r["model"]:<18}{r["accuracy"]:<12.4f}{r["precision"]:<12.4f}'
              f'{r["recall"]:<10.4f}{r["f1_score"]:<10.4f}'
              f'{r["inference_ms_per_image"]:<10.2f}{r["model_size_mb"]:<10.1f}')
    print('='*90)
    
    # График сравнения
    valid = [r for r in results if r is not None]
    names = [r['model'] for r in valid]
    accs = [r['accuracy'] for r in valid]
    f1s = [r['f1_score'] for r in valid]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    ax.bar(x - 0.2, accs, 0.4, label='Accuracy')
    ax.bar(x + 0.2, f1s, 0.4, label='F1-score')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel('Метрика')
    ax.set_title('Сравнение моделей')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparison.png'), dpi=100)
    plt.close()
    
    save_json(valid, os.path.join(RESULTS_DIR, 'comparison.json'))


if __name__ == '__main__':
    _, _, test_loader = get_dataloaders(batch_size=64)
    
    models_list = ['SimpleCNN', 'ResNet18', 'MobileNetV2', 'EfficientNetB0', 'VGG11']
    results = []
    for name in models_list:
        results.append(evaluate_model(name, test_loader))
    
    make_comparison_table(results)
    
    # Выбираем лучшую
    valid = [r for r in results if r is not None]
    best = max(valid, key=lambda r: r['f1_score'])
    print(f'\n🏆 Лучшая модель: {best["model"]} (F1={best["f1_score"]:.4f})')
    save_json({'best_model': best['model']}, os.path.join(RESULTS_DIR, 'best_model.json'))