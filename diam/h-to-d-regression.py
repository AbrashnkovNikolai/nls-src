# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
import warnings
import os
import json
from pathlib import Path
import seaborn as sns
from collections import defaultdict, Counter

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ======================
# 1. НАСТРОЙКИ И КОНСТАНТЫ
# ======================
print("\n" + "=" * 100)
print("УЛУЧШЕННАЯ ОБРАБОТКА С ВАЛИДАЦИЕЙ МОДЕЛЕЙ")
print("=" * 100)

DATA_FOLDER = 'csv_etalons'
MIN_SAMPLES = 10


# Функции моделей с ограничениями для реалистичности
def power_func(x, a, b, c):
    """Степенная модель с ограничениями"""
    # Ограничиваем параметры для реалистичности
    b = np.clip(b, 0.1, 3.0)  # Степень от 0.1 до 3
    return a * (x ** b) + c


def exponential_func(x, a, b, c):
    """Экспоненциальная модель с ограничениями"""
    # b обычно отрицательный для деревьев
    b = np.clip(b, -20, -0.1)
    return a * np.exp(b * x) + c


def quadratic_func(x, a, b, c):
    """Квадратичная модель"""
    return a * x ** 2 + b * x + c


def linear_func(x, a, b):
    """Линейная модель"""
    return a * x + b


# ======================
# 2. ФУНКЦИИ ДЛЯ ОБРАБОТКИ
# ======================
def normalize_species_name(species):
    """Нормализация названия породы"""
    if pd.isna(species):
        return None

    species_str = str(species).strip()

    if species_str.endswith('.0'):
        species_str = species_str[:-2]

    try:
        num = float(species_str)
        if num.is_integer():
            return str(int(num))
    except:
        pass

    return species_str


def is_model_realistic(params, model_type):
    """Проверка реалистичности параметров модели"""
    if model_type == 'power':
        a, b, c = params
        # Проверяем, что модель не вырождена
        if abs(b) < 0.001:  # Слишком маленькая степень
            return False
        if abs(a) > 10000:  # Слишком большой коэффициент
            return False
        return True

    elif model_type == 'exponential':
        a, b, c = params
        # b должен быть отрицательным для деревьев
        if b >= -0.1:
            return False
        if abs(a) > 1000:
            return False
        return True

    elif model_type == 'quadratic':
        a, b, c = params
        # Проверяем, что парабола не слишком крутая
        if abs(a) > 500:
            return False
        return True

    elif model_type == 'linear':
        a, b = params
        if abs(a) > 200:  # Слишком крутой наклон
            return False
        return True

    return True


def fit_model_with_validation(X, y, species_name):
    """Обучение модели с валидацией и проверкой реалистичности"""
    if len(X) < MIN_SAMPLES * 2:
        return None, None, None

    models_to_test = [
        ('quadratic', quadratic_func, 3, [-100, 100, 0]),
        ('power', power_func, 3, [50, 1, 0]),
        ('linear', linear_func, 2, [50, 5]),
        ('exponential', exponential_func, 3, [-50, -3, 25]),
    ]

    best_score = -float('inf')
    best_model_name = None
    best_params = None

    # Разделяем на обучение и тест
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(0.65 * len(X))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    for model_name, model_func, n_params, p0 in models_to_test:
        try:
            if n_params == 2:
                params, _ = curve_fit(model_func, X_train, y_train, p0=p0, maxfev=10000,
                                      bounds=([-200, -50], [200, 100]))
                y_pred = model_func(X_test, *params)
            else:
                if model_name == 'power':
                    bounds = ([0.1, 0.1, -50], [200, 3, 100])
                elif model_name == 'exponential':
                    bounds = ([-1000, -20, -50], [0, -0.1, 100])
                else:
                    bounds = ([-500, -200, -50], [500, 200, 100])

                params, _ = curve_fit(model_func, X_train, y_train, p0=p0, maxfev=10000,
                                      bounds=bounds)
                y_pred = model_func(X_test, *params)

            # Проверяем на NaN и реалистичность
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                continue

            if not is_model_realistic(params, model_name):
                continue

            # Проверяем, что предсказания в разумных пределах
            if np.any(y_pred < 0) or np.any(y_pred > 100):
                continue

            r2 = r2_score(y_test, y_pred)

            if r2 > best_score:
                best_score = r2
                best_model_name = model_name
                best_params = params

        except Exception as e:
            continue

    return best_model_name, best_params, best_score


# ======================
# 3. ОБРАБОТКА ДАННЫХ
# ======================
print(f"\n📁 Загружаем данные из: {DATA_FOLDER}")

if not os.path.exists(DATA_FOLDER):
    print("❌ Папка не найдена!")
    exit()

csv_files = list(Path(DATA_FOLDER).glob('*.csv'))
print(f"✅ Найдено {len(csv_files)} файлов")

# Собираем все данные
all_data = defaultdict(list)

for file_path in csv_files:
    print(f"\n📄 Обработка: {file_path.name}")

    try:
        df = pd.read_csv(file_path,sep=';')

        # Проверяем колонки
        required = ['d', 'h', 's']
        available = set(df.columns)

        col_mapping = {}
        for req in required:
            if req in available:
                col_mapping[req] = req
            else:
                # Ищем похожие
                for col in available:
                    if req in col.lower() or col.lower() in req:
                        col_mapping[req] = col
                        break

        if len(col_mapping) < 3:
            print(f"  ⚠️  Пропускаем: не все колонки найдены")
            continue

        df = df.rename(columns=col_mapping)
        df['s'] = df['s'].apply(normalize_species_name)
        df = df[df['s'].notna()]

        # Преобразуем и фильтруем
        df['d'] = pd.to_numeric(df['d'], errors='coerce')
        df['h'] = pd.to_numeric(df['h'], errors='coerce')
        df = df.dropna(subset=['d', 'h', 's'])

        df = df[(df['d'] >= 0.05) & (df['d'] <= 0.6)]
        df = df[(df['h'] >= 1.0) & (df['h'] <= 50.0)]

        if len(df) == 0:
            print(f"  ⚠️  Нет данных после фильтрации")
            continue

        # Группируем по породам
        for species in df['s'].unique():
            species_data = df[df['s'] == species]
            if len(species_data) >= MIN_SAMPLES:
                all_data[species].append({
                    'file': file_path.name,
                    'data': species_data
                })

        print(f"  ✅ Загружено {len(df)} строк")

    except Exception as e:
        print(f"  ❌ Ошибка: {e}")

# ======================
# 4. ПОСТРОЕНИЕ МОДЕЛЕЙ
# ======================
print("\n" + "=" * 100)
print("ПОСТРОЕНИЕ УЛУЧШЕННЫХ МОДЕЛЕЙ")
print("=" * 100)

all_models = {}
species_summary = []
species_raw_data = {}  # Для визуализации

for species, file_data_list in all_data.items():
    if len(file_data_list) == 0:
        continue

    # Объединяем данные
    combined_dfs = []
    for item in file_data_list:
        combined_dfs.append(item['data'])

    df_combined = pd.concat(combined_dfs, ignore_index=True)
    total_samples = len(df_combined)
    files_used = list(set(item['file'] for item in file_data_list))

    if total_samples < MIN_SAMPLES * 2:
        print(f"\n🌲 {species}: недостаточно данных ({total_samples} < {MIN_SAMPLES * 2})")
        continue

    print(f"\n🌲 ПОРДА: {species}")
    print(f"   📊 Образцов: {total_samples} из {len(files_used)} файлов")
    print(f"   📏 Диаметр: {df_combined['d'].min():.3f} - {df_combined['d'].max():.3f} м")
    print(f"   📐 Высота: {df_combined['h'].min():.1f} - {df_combined['h'].max():.1f} м")

    X = df_combined['d'].values
    y = df_combined['h'].values

    # Сохраняем сырые данные для визуализации
    species_raw_data[species] = {
        'X': X,
        'y': y,
        'df': df_combined
    }

    # Обучаем модель
    model_name, params, cv_score = fit_model_with_validation(X, y, species)

    if model_name and params is not None:
        try:
            # Переобучаем на всех данных
            if len(params) == 2:
                final_params, _ = curve_fit(
                    globals()[f"{model_name}_func"], X, y, p0=params, maxfev=10000
                )
            else:
                final_params, _ = curve_fit(
                    globals()[f"{model_name}_func"], X, y, p0=params, maxfev=10000
                )

            # Проверяем реалистичность
            if not is_model_realistic(final_params, model_name):
                print(f"   ⚠️  Модель нереалистична, пробуем другую...")
                # Пробуем линейную модель как запасной вариант
                try:
                    final_params, _ = curve_fit(linear_func, X, y, p0=[50, 5], maxfev=5000)
                    model_name = 'linear'
                    if not is_model_realistic(final_params, model_name):
                        continue
                except:
                    continue

            # Рассчитываем метрики
            if model_name == 'linear':
                y_pred = linear_func(X, *final_params)
            elif model_name == 'quadratic':
                y_pred = quadratic_func(X, *final_params)
            elif model_name == 'power':
                y_pred = power_func(X, *final_params)
            elif model_name == 'exponential':
                y_pred = exponential_func(X, *final_params)

            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)

            # Создаем уравнение
            if model_name == 'quadratic':
                a, b, c = final_params
                equation = f"h = {a:.1f}·d² + {b:.1f}·d + {c:.1f}"
            elif model_name == 'power':
                a, b, c = final_params
                equation = f"h = {a:.1f}·d^{b:.2f} + {c:.1f}"
            elif model_name == 'linear':
                a, b = final_params
                equation = f"h = {a:.1f}·d + {b:.1f}"
            elif model_name == 'exponential':
                a, b, c = final_params
                equation = f"h = {a:.1f}·e^({b:.2f}·d) + {c:.1f}"

            print(f"   🎯 Модель: {model_name}")
            print(f"   📐 {equation}")
            print(f"   📊 R²: {r2:.4f} (валидация: {cv_score:.4f})")
            print(f"   📏 RMSE: {rmse:.2f} м")
            print(f"   📏 MAE: {mae:.2f} м")

            # Сохраняем модель
            all_models[species] = {
                'model_type': model_name,
                'equation': equation,
                'parameters': [float(p) for p in final_params],
                'metrics': {
                    'r2': float(r2),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'cv_score': float(cv_score)
                },
                'samples': total_samples,
                'files': files_used,
                'diameter_range': [float(df_combined['d'].min()), float(df_combined['d'].max())],
                'height_range': [float(df_combined['h'].min()), float(df_combined['h'].max())]
            }

            species_summary.append({
                'Порода': species,
                'Образцы': total_samples,
                'Файлы': len(files_used),
                'Модель': model_name,
                'R²': r2,
                'RMSE': rmse,
                'Уравнение': equation,
                'Диапазон_d': f"{df_combined['d'].min():.3f}-{df_combined['d'].max():.3f}",
                'Диапазон_h': f"{df_combined['h'].min():.1f}-{df_combined['h'].max():.1f}"
            })

        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    else:
        print(f"   ❌ Не удалось построить модель")

# ======================
# 5. АНАЛИЗ И СОХРАНЕНИЕ
# ======================
print("\n" + "=" * 100)
print("АНАЛИЗ РЕЗУЛЬТАТОВ")
print("=" * 100)

if not all_models:
    print("❌ Не построено моделей!")
    exit()

# Создаем DataFrame
summary_df = pd.DataFrame(species_summary)
summary_df = summary_df.sort_values('Образцы', ascending=False)

print(f"\n✅ Построено моделей: {len(all_models)}")
print(f"✅ Всего образцов: {summary_df['Образцы'].sum()}")

print("\n📊 СВОДНАЯ ТАБЛИЦА:")
print("-" * 120)
print(
    f"{'Порода':<6} {'Образцы':<8} {'Модель':<12} {'R²':<8} {'RMSE':<8} {'Диапазон d (м)':<20} {'Диапазон h (м)':<20}")
print("-" * 120)

for _, row in summary_df.iterrows():
    print(f"{row['Порода']:<6} {row['Образцы']:<8} {row['Модель']:<12} "
          f"{row['R²']:<8.3f} {row['RMSE']:<8.2f} {row['Диапазон_d']:<20} {row['Диапазон_h']:<20}")

print("-" * 120)

# Сохраняем модели
with open('improved_species_models.json', 'w', encoding='utf-8') as f:
    json.dump(all_models, f, indent=2, ensure_ascii=False)
print("\n✓ Модели сохранены в 'improved_species_models.json'")

summary_df.to_csv('improved_models_summary.csv', index=False, encoding='utf-8-sig')
print("✓ Сводная таблица сохранена в 'improved_models_summary.csv'")

# ======================
# 6. ВИЗУАЛИЗАЦИЯ
# ======================
print("\n" + "=" * 100)
print("ВИЗУАЛИЗАЦИЯ МОДЕЛЕЙ")
print("=" * 100)

if all_models and species_raw_data:
    # Сортируем породы по количеству образцов
    species_sorted = sorted(all_models.keys(),
                            key=lambda s: all_models[s]['samples'],
                            reverse=True)

    # Ограничиваем количество графиков
    n_to_plot = min(9, len(species_sorted))

    n_cols = 3
    n_rows = (n_to_plot + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    if n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    elif hasattr(axes, 'ravel'):
        axes = axes.ravel()
    else:
        axes = [axes]

    for idx, species in enumerate(species_sorted[:n_to_plot]):
        if idx >= len(axes):
            break

        ax = axes[idx]
        model_info = all_models[species]
        raw_data = species_raw_data[species]

        # Данные
        X = raw_data['X']
        y = raw_data['y']

        ax.scatter(X, y, alpha=0.3, s=10, label=f"Данные (n={len(X)})")

        # Кривая модели
        x_plot = np.linspace(max(0.05, X.min()), min(0.6, X.max()), 100)
        params = model_info['parameters']

        if model_info['model_type'] == 'quadratic':
            y_plot = quadratic_func(x_plot, *params)
        elif model_info['model_type'] == 'power':
            y_plot = power_func(x_plot, *params)
        elif model_info['model_type'] == 'linear':
            y_plot = linear_func(x_plot, *params)
        elif model_info['model_type'] == 'exponential':
            y_plot = exponential_func(x_plot, *params)

        ax.plot(x_plot, y_plot, 'r-', linewidth=2, label='Модель')

        # Информация
        info_text = f"Порода {species}\\n"
        info_text += f"{model_info['model_type']}\\n"
        info_text += f"R²={model_info['metrics']['r2']:.3f}\\n"
        info_text += f"RMSE={model_info['metrics']['rmse']:.1f} м"

        ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Диаметр (м)', fontsize=10)
        ax.set_ylabel('Высота (м)', fontsize=10)
        ax.set_title(f'Порода {species}', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Скрываем пустые subplots
    for idx in range(len(species_sorted[:n_to_plot]), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('improved_models_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✓ Графики сохранены в 'improved_models_visualization.png'")


# ======================
# 8. ПРАКТИЧЕСКИЕ ПРОГНОЗЫ
# ======================
print("\n" + "=" * 100)
print("ПРАКТИЧЕСКИЕ ПРОГНОЗЫ")
print("=" * 100)

print("\n📋 ТАБЛИЦА ПРОГНОЗОВ ДЛЯ ОСНОВНЫХ ПОРОД:")
print("-" * 80)
print(f"{'Порода':<6} {'Диаметр':<15} {'Высота':<15} {'±Погр.':<10} {'Модель':<12} {'R²':<8}")
print("-" * 80)

# Тестовые диаметры
test_diameters = [0.1, 0.2, 0.3, 0.4, 0.5]

# Берем топ-5 пород по количеству данных
top_species = summary_df.head(5)['Порода'].tolist()

for species in top_species:
    if species in all_models:
        model = all_models[species]
        for d in test_diameters:
            # Проверяем диапазон
            d_min, d_max = model['diameter_range']
            if d < d_min or d > d_max:
                continue

            # Вычисляем высоту
            params = model['parameters']
            if model['model_type'] == 'quadratic':
                a, b, c = params
                height = a * d ** 2 + b * d + c
            elif model['model_type'] == 'power':
                a, b, c = params
                height = a * (d ** b) + c
            elif model['model_type'] == 'linear':
                a, b = params
                height = a * d + b
            elif model['model_type'] == 'exponential':
                a, b, c = params
                height = a * np.exp(b * d) + c

            uncertainty = model['metrics']['rmse']

            print(f"{species:<6} {d:.2f} м ({d * 100:3.0f} см) {height:7.1f} м {uncertainty:6.1f} м "
                  f"{model['model_type']:<12} {model['metrics']['r2']:6.3f}")
    print("-" * 80)

print("\n" + "=" * 100)
print("РАБОТА ЗАВЕРШЕНА!")
print("=" * 100)