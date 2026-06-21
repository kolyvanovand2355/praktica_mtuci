import os
import io
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from torchvision import transforms

from utils import DEVICE, IMG_SIZE, CLASS_NAMES, load_json
from train_models import get_model

MODELS_DIR = 'models'
RESULTS_DIR = 'results'


st.set_page_config(
    page_title='Распознавание дорожных знаков',
    layout='wide'
)
st.markdown("""
    <style>
    
    .stDeployButton {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    
    
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}
    
    
    [data-testid="stHeaderActionElements"] {display: none !important;}
    h1 > div > a, h2 > div > a, h3 > div > a,
    h4 > div > a, h5 > div > a, h6 > div > a {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name):
    """Загружаем модель один раз и кэшируем"""
    model = get_model(model_name).to(DEVICE)
    weights_path = os.path.join(MODELS_DIR, f'{model_name}.pt')
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


@st.cache_data
def get_best_model_name():
    """Получаем имя лучшей модели из JSON"""
    try:
        info = load_json(os.path.join(RESULTS_DIR, 'best_model.json'))
        return info['best_model']
    except FileNotFoundError:
        return 'EfficientNetB0'


def get_available_models():
    """Список всех доступных обученных моделей"""
    if not os.path.exists(MODELS_DIR):
        return []
    models = [f.replace('.pt', '') for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]
    return sorted(models)



transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict(model, img):
    
    x = transform(img).unsqueeze(0).to(DEVICE)
    start = time.time()
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)[0]
    elapsed_ms = (time.time() - start) * 1000
    return probs.cpu().numpy(), elapsed_ms



st.sidebar.title('Настройки')

available = get_available_models()
if not available:
    st.error(' Не найдено ни одной обученной модели в папке `models/`. '
             'Сначала запустите `python train_models.py`')
    st.stop()

best_name = get_best_model_name()
default_idx = available.index(best_name) if best_name in available else 0

selected_model = st.sidebar.selectbox(
    'Выберите модель:',
    available,
    index=default_idx,
    help='По умолчанию выбрана лучшая модель по F1-score'
)

st.sidebar.markdown('---')
st.sidebar.markdown(f'**Устройство:** `{DEVICE}`')
st.sidebar.markdown(f'**Размер входа:** `{IMG_SIZE}×{IMG_SIZE}`')
st.sidebar.markdown(f'**Кол-во классов:** `{len(CLASS_NAMES)}`')

# Показываем метрики лучшей модели если есть
eval_path = os.path.join(RESULTS_DIR, f'{selected_model}_eval.json')
if os.path.exists(eval_path):
    eval_info = load_json(eval_path)
    st.sidebar.markdown('---')
    st.sidebar.markdown('Метрики модели')
    st.sidebar.metric('Accuracy', f'{eval_info["accuracy"]:.4f}')
    st.sidebar.metric('F1-score', f'{eval_info["f1_score"]:.4f}')
    st.sidebar.metric('Inference', f'{eval_info["inference_ms_per_image"]:.2f} мс')
    st.sidebar.metric('Размер модели', f'{eval_info["model_size_mb"]:.1f} МБ')


# ====== ОСНОВНАЯ СТРАНИЦА ======
st.title('🚦 Распознавание дорожных знаков')
st.markdown(f'**Текущая модель:** `{selected_model}` | '
            f'**Устройство:** `{DEVICE}`')

# Загружаем модель
with st.spinner(f'Загружаем модель {selected_model}...'):
    model = load_model(selected_model)

st.markdown('---')

# Создаём вкладки
tab1, tab2, tab3 = st.tabs(['Распознавание', 'Массовая обработка', 'О проекте'])


with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader('Загрузка изображения')
        uploaded = st.file_uploader(
            'Выберите изображение знака',
            type=['png', 'jpg', 'jpeg', 'ppm', 'bmp'],
            key='single'
        )
        
        if uploaded is not None:
            img = Image.open(uploaded).convert('RGB')
            st.image(img, caption=f'Загружено: {uploaded.name}', use_container_width=True)
    
    with col2:
        st.subheader('Результат распознавания')
        if uploaded is not None:
            probs, inf_time = predict(model, img)
            
            # Топ-1
            top1_idx = int(np.argmax(probs))
            top1_conf = float(probs[top1_idx])
            
            # Цвет в зависимости от уверенности
            if top1_conf > 0.8:
                st.success(f':blue[ Знак: **{CLASS_NAMES[top1_idx]}**]')
            elif top1_conf > 0.5:
                st.warning(f':orange[  Знак: **{CLASS_NAMES[top1_idx]}**] ')
            else:
                st.error(f':orange[  Знак: **{CLASS_NAMES[top1_idx]}**] ')
            
            # Метрики
            mcol1, mcol2 = st.columns(2)
            mcol1.metric('Уверенность', f'{top1_conf*100:.1f}%')
            mcol2.metric('Время обработки', f'{inf_time:.2f} мс')
            
            # Прогресс-бар
            st.progress(top1_conf)
            
            # Топ-5 предсказаний
            st.markdown('### Топ-5 предсказаний:')
            top5_idx = np.argsort(probs)[::-1][:5]
            top5_data = pd.DataFrame({
                'Класс': [CLASS_NAMES[i] for i in top5_idx],
                'Уверенность': [f'{probs[i]*100:.2f}%' for i in top5_idx],
                'Вероятность': [float(probs[i]) for i in top5_idx]
            })
           

            st.dataframe(
                top5_data[['Класс', 'Уверенность']],
                use_container_width=True,
                hide_index=True
            )
            
            # График
            st.bar_chart(
                top5_data.set_index('Класс')['Вероятность'],
                use_container_width=True
            )
        else:
            st.info('Загрузите изображение знака с помощью кнопки слева')

# ============ ВКЛАДКА 2: НЕСКОЛЬКО КАРТИНОК ============
with tab2:
    st.subheader('Массовая обработка изображений')
    st.markdown('Загрузите несколько изображений сразу для массового распознавания.')
    
    uploaded_batch = st.file_uploader(
        'Выберите несколько изображений',
        type=['png', 'jpg', 'jpeg', 'ppm', 'bmp'],
        accept_multiple_files=True,
        key='batch'
    )
    
    if uploaded_batch:
        st.markdown(f'**Загружено файлов:** {len(uploaded_batch)}')
        
        if st.button('Распознать все', type='primary'):
            results = []
            progress = st.progress(0)
            status = st.empty()
            
            for i, file in enumerate(uploaded_batch):
                status.text(f'Обработка {i+1}/{len(uploaded_batch)}: {file.name}')
                img = Image.open(file).convert('RGB')
                probs, inf_time = predict(model, img)
                top1_idx = int(np.argmax(probs))
                results.append({
                    'Файл': file.name,
                    'Класс': CLASS_NAMES[top1_idx],
                    'Уверенность': f'{probs[top1_idx]*100:.2f}%',
                    'Время (мс)': f'{inf_time:.2f}'
                })
                progress.progress((i + 1) / len(uploaded_batch))
            
            status.text('Готово!')
            
            df_results = pd.DataFrame(results)
            st.markdown('Результаты:')
            st.dataframe(df_results, use_container_width=True, hide_index=True)
            
            # Скачать CSV
            csv = df_results.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                'Скачать результаты в CSV формате)',
                csv,
                file_name='predictions.csv',
                mime='text/csv'
            )
            
            
            st.markdown('Превью изображений:')
            cols = st.columns(4)
            for i, file in enumerate(uploaded_batch[:12]):
                with cols[i % 4]:
                    img = Image.open(file).convert('RGB')
                    st.image(img, caption=results[i]['Класс'], use_container_width=True)


with tab3:
    st.subheader('О проекте')
    st.markdown("""
    ### Назначение
    Распознавания дорожных знаков на базе нейронных сетей.
    
    ### Датасет
    **GTSRB** — используется немецкий датасет дорожных знаков: 
    - 43 класса дорожных знаков
    - ~39 000 обучающих изображений
    - ~12 600 тестовых изображений
    
    ### Архитектуры
    В проекте обучены и сравнены 5 моделей:
    1. **SimpleCNN** — своя простая свёрточная сеть
    2. **ResNet18** — остаточная сеть
    3. **MobileNetV2** — лёгкая мобильная сеть
    4. **EfficientNet-B0** — эффективная масштабируемая сеть (лучшая)
    5. **VGG11** — классическая глубокая сеть
    
    ### Метрики оценки
    - Accuracy, Precision, Recall, F1-score
    - Confusion matrix
    - Время инференса
    - Размер модели
    
    ### Технологии
    - **PyTorch** — фреймворк глубокого обучения
    - **Streamlit** — веб-интерфейс
    - **scikit-learn** — метрики качества
    """)
    
    
    comparison_path = os.path.join(RESULTS_DIR, 'comparison.json')
    if os.path.exists(comparison_path):
        st.markdown('### Сравнение моделей')
        comp = load_json(comparison_path)
        df_comp = pd.DataFrame([{
            'Модель': r['model'],
            'Accuracy': f'{r["accuracy"]:.4f}',
            'Precision': f'{r["precision"]:.4f}',
            'Recall': f'{r["recall"]:.4f}',
            'F1-score': f'{r["f1_score"]:.4f}',
            'Inference (мс)': f'{r["inference_ms_per_image"]:.2f}',
            'Размер (МБ)': f'{r["model_size_mb"]:.1f}'
        } for r in comp])
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    
    comp_img = os.path.join(RESULTS_DIR, 'comparison.png')
    if os.path.exists(comp_img):
        st.image(comp_img, caption='Сравнение моделей по метрикам',
                 use_container_width=True)


st.markdown('---')
st.markdown(
    "<p style='text-align: center; color: red;'>"
    "Распознавание дорожных знаков,выполнил: "
    "Колыванов Андрей Николаевич"
    "</p>",
    unsafe_allow_html=True
)