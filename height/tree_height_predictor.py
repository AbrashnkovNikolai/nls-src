"""
Модуль для прогнозирования высоты деревьев разных пород по диаметру
Использует экспоненциальные модели, обученные отдельно для каждой породы
"""

import numpy as np
import json

class TreeHeightPredictor:
    """Класс для прогнозирования высоты деревьев"""

    def __init__(self, models_file='exponential_models_summary.json'):
        """
        Инициализация прогнозировщика

        Parameters:
        -----------
        models_file : str
            Путь к файлу с параметрами моделей
        """
        self.models = self._load_models(models_file)
        self.species_list = list(self.models.keys())

    def _load_models(self, models_file):
        """Загрузка моделей из JSON файла"""
        with open(models_file, 'r', encoding='utf-8') as f:
            models = json.load(f)
        return models

    def predict(self, species, diameter):
        """
        Прогнозирование высоты дерева

        Parameters:
        -----------
        species : str
            Порода дерева
        diameter : float
            Диаметр ствола в см

        Returns:
        --------
        height : float
            Предсказанная высота в метрах
        """
        if species not in self.models:
            raise ValueError(
                f"Порода '{species}' не найдена. "
                f"Доступные породы: {', '.join(self.species_list)}"
            )

        # Получаем параметры модели
        params = self.models[species]['parameters']
        a, b, c = params['a'], params['b'], params['c']

        # Экспоненциальная модель: y = a * exp(b * x) + c
        height = a * np.exp(b * diameter) + c

        return height

    def predict_batch(self, species, diameters):
        """
        Прогнозирование высоты для нескольких диаметров

        Parameters:
        -----------
        species : str
            Порода дерева
        diameters : list or np.array
            Массив диаметров в см

        Returns:
        --------
        heights : np.array
            Массив предсказанных высот в метрах
        """
        if species not in self.models:
            raise ValueError(
                f"Порода '{species}' не найдена. "
                f"Доступные породы: {', '.join(self.species_list)}"
            )

        diameters = np.array(diameters)
        params = self.models[species]['parameters']
        a, b, c = params['a'], params['b'], params['c']

        heights = a * np.exp(b * diameters) + c

        return heights

    def get_species_list(self):
        """Получить список доступных пород"""
        return self.species_list

    def get_model_info(self, species):
        """Получить информацию о модели для конкретной породы"""
        if species not in self.models:
            raise ValueError(f"Порода '{species}' не найдена")

        return self.models[species]

# Пример использования
if __name__ == "__main__":
    # Создаем прогнозировщик
    predictor = TreeHeightPredictor()

    # Получаем список доступных пород
    print("Доступные породы деревьев:")
    for species in predictor.get_species_list():
        print(f"  - {species}")

    # Примеры прогнозов
    print("\nПримеры прогнозов высоты:")
    print("-" * 60)

    test_cases = [
        ("ель", 0.3),
        ("сосна", 0.4),
        ("береза", 0.25),
        ("дуб", 0.5)
    ]

    for species, diameter in test_cases:
        try:
            height = predictor.predict(species, diameter)
            model_info = predictor.get_model_info(species)
            equation = model_info.get('equation', 'N/A')

            print(f"Порода: {species}")
            print(f"Диаметр: {diameter:.2f} см")
            print(f"Предсказанная высота: {height:.2f} м")
            print(f"Модель: {equation}")
            print(f"Данных для обучения: {model_info.get('data_size', 'N/A')} деревьев")
            print("-" * 60)
        except ValueError as e:
            print(f"Ошибка для породы '{species}': {e}")
            print("-" * 60)

    # Прогноз для нескольких диаметров
    print("\nПрогноз для сосны при разных диаметрах:")
    diameters = [0.2, 0.3, 0.4, 0.5, 0.6]

    try:
        heights = predictor.predict_batch("сосна", diameters)
        for d, h in zip(diameters, heights):
            print(f"  Диаметр {d:.2f} см -> Высота {h:.2f} м")
    except ValueError as e:
        print(f"Ошибка: {e}")
