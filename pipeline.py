"""
Pipeline для классификации мобильных телефонов по ценовому диапазону
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


class MobilePricePipeline:
    """Класс для создания и управления pipeline классификации"""
    
    def __init__(self, model_type='random_forest'):
        """
        Инициализация pipeline
        
        Args:
            model_type: тип модели ('random_forest', 'logistic', 'svc', 'tree')
        """
        self.model_type = model_type
        self.pipeline = self._create_pipeline()
        
    def _create_pipeline(self):
        """Создание pipeline с предобработкой и моделью"""
        
        # Выбор модели
        if self.model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
        elif self.model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'svc':
            model = SVC(random_state=42)
        elif self.model_type == 'tree':
            model = DecisionTreeClassifier(random_state=42)
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
        
        # Создание pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        return pipeline
    
    def train(self, X_train, y_train):
        """Обучение модели"""
        self.pipeline.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Предсказание"""
        return self.pipeline.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Оценка модели"""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"\nМодель: {self.model_type}")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return accuracy
    
    def save(self, filename):
        """Сохранение pipeline"""
        with open(filename, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Pipeline сохранен в {filename}")
    
    @staticmethod
    def load(filename):
        """Загрузка pipeline"""
        with open(filename, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"Pipeline загружен из {filename}")
        return pipeline


def load_data():
    """Загрузка данных"""
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    # Разделение на признаки и целевую переменную
    X_train = train_df.drop('price_range', axis=1)
    y_train = train_df['price_range']
    
    # Для тестовых данных (если есть price_range)
    if 'price_range' in test_df.columns:
        X_test = test_df.drop('price_range', axis=1)
        y_test = test_df['price_range']
    else:
        X_test = test_df
        y_test = None
    
    return X_train, y_train, X_test, y_test


def main():
    """Основная функция"""
    print("Загрузка данных...")
    X_train, y_train, X_test, y_test = load_data()
    
    # Разделение train на train и validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nРазмер обучающей выборки: {X_tr.shape}")
    print(f"Размер валидационной выборки: {X_val.shape}")
    
    # Обучение разных моделей
    models = ['random_forest', 'logistic', 'svc', 'tree']
    results = {}
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Обучение модели: {model_type}")
        print('='*50)
        
        # Создание и обучение pipeline
        pipeline = MobilePricePipeline(model_type=model_type)
        pipeline.train(X_tr, y_tr)
        
        # Оценка на валидационной выборке
        accuracy = pipeline.evaluate(X_val, y_val)
        results[model_type] = accuracy
        
        # Сохранение модели
        filename = f'pipeline_{model_type}.pkl'
        pipeline.save(filename)
    
    # Вывод результатов
    print(f"\n{'='*50}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print('='*50)
    for model_type, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_type:20s}: {accuracy:.4f}")
    
    # Лучшая модель
    best_model = max(results, key=results.get)
    print(f"\nЛучшая модель: {best_model} (Accuracy: {results[best_model]:.4f})")


if __name__ == '__main__':
    main()
