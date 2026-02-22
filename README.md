# Mobile Price Classification Pipeline

Проект для классификации мобильных телефонов по ценовому диапазону с использованием sklearn Pipeline.

## Структура проекта

```
.
├── data/
│   ├── train.csv          # Обучающие данные
│   └── test.csv           # Тестовые данные
├── pipeline.py            # Основной модуль с Pipeline
├── use_pipeline.py        # Скрипт для использования обученной модели
├── mobile.ipynb           # Jupyter notebook с анализом
├── task1.ipynb            # Jupyter notebook с задачами
└── README.md              # Документация
```

## Установка зависимостей

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Использование

### 1. Обучение моделей

Запустите скрипт для обучения всех моделей:

```bash
python pipeline.py
```

Это создаст следующие файлы:
- `pipeline_random_forest.pkl` - Random Forest модель
- `pipeline_logistic.pkl` - Logistic Regression модель
- `pipeline_svc.pkl` - Support Vector Classifier модель
- `pipeline_tree.pkl` - Decision Tree модель

### 2. Использование обученной модели

```bash
python use_pipeline.py
```

### 3. Программное использование

```python
from pipeline import MobilePricePipeline

# Создание и обучение pipeline
pipeline = MobilePricePipeline(model_type='random_forest')
pipeline.train(X_train, y_train)

# Предсказание
predictions = pipeline.predict(X_test)

# Оценка
accuracy = pipeline.evaluate(X_test, y_test)

# Сохранение
pipeline.save('my_model.pkl')

# Загрузка
loaded_pipeline = MobilePricePipeline.load('my_model.pkl')
```

## Описание Pipeline

Pipeline включает следующие этапы:

1. **StandardScaler** - нормализация признаков
2. **Classifier** - модель классификации (Random Forest, Logistic Regression, SVC, Decision Tree)

## Признаки датасета

- `battery_power` - мощность батареи
- `blue` - наличие Bluetooth
- `clock_speed` - тактовая частота процессора
- `dual_sim` - поддержка двух SIM-карт
- `fc` - фронтальная камера (мегапиксели)
- `four_g` - поддержка 4G
- `int_memory` - внутренняя память (ГБ)
- `m_dep` - толщина мобильного телефона (см)
- `mobile_wt` - вес телефона
- `n_cores` - количество ядер процессора
- `pc` - основная камера (мегапиксели)
- `px_height` - высота разрешения экрана
- `px_width` - ширина разрешения экрана
- `ram` - оперативная память (МБ)
- `sc_h` - высота экрана (см)
- `sc_w` - ширина экрана (см)
- `talk_time` - время разговора
- `three_g` - поддержка 3G
- `touch_screen` - наличие сенсорного экрана
- `wifi` - наличие Wi-Fi

## Целевая переменная

`price_range` - ценовой диапазон (0, 1, 2, 3)

## Преимущества использования Pipeline

1. **Упрощение кода** - все этапы обработки в одном объекте
2. **Предотвращение утечки данных** - scaler обучается только на train данных
3. **Удобство развертывания** - один файл для сохранения всей модели
4. **Воспроизводимость** - гарантия одинаковой обработки данных
5. **Легкость использования** - простой интерфейс для предсказаний

## Автор

Проект создан для классификации мобильных телефонов по ценовым категориям.
