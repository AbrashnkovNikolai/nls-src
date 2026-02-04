import numpy as np
import matplotlib.pyplot as plt
import laspy
from tqdm import tqdm


def plot_color_histogram_stacked(las_path: str, bins: int = 256, chunk_size: int = 50_000_000):
    """
    Строит гистограмму с наслаивающимися цветами (аддитивное смешение RGB).

    Параметры:
        las_path: путь к LAS/LAZ файлу
        bins: количество бинов для гистограммы
        chunk_size: размер чанка для чтения (точек)
    """
    # Инициализация счетчиков для RGB
    red_counts = np.zeros(bins, dtype=np.int64)
    green_counts = np.zeros(bins, dtype=np.int64)
    blue_counts = np.zeros(bins, dtype=np.int64)

    try:
        with laspy.open(las_path) as las:
            total_points = las.header.point_count

            for chunk in tqdm(las.chunk_iterator(chunk_size),
                              total=int(np.ceil(total_points / chunk_size)),
                              desc="Обработка точек"):
                if hasattr(chunk, 'red'):
                    red = chunk.red >> 8
                    green = chunk.green >> 8
                    blue = chunk.blue >> 8

                    red_hist, _ = np.histogram(red, bins=bins, range=(0, 255))
                    green_hist, _ = np.histogram(green, bins=bins, range=(0, 255))
                    blue_hist, _ = np.histogram(blue, bins=bins, range=(0, 255))

                    red_counts += red_hist
                    green_counts += green_hist
                    blue_counts += blue_hist

        # Нормализация для визуализации (0-1)
        max_count = max(np.max(red_counts), np.max(green_counts), np.max(blue_counts))
        red_norm = red_counts / max_count
        green_norm = green_counts / max_count
        blue_norm = blue_counts / max_count

        # Создание RGB массива для каждого бина
        rgb_colors = np.zeros((bins, 3))
        rgb_colors[:, 0] = red_norm  # R канал
        rgb_colors[:, 1] = green_norm  # G канал
        rgb_colors[:, 2] = blue_norm  # B канал

        # Ограничение значений (0-1)
        rgb_colors = np.clip(rgb_colors, 0, 1)

        # Построение гистограммы
        plt.figure(figsize=(14, 7))
        x = np.arange(bins)

        # Рисуем столбцы с комбинированными цветами
        bars = plt.bar(x, np.ones(bins), width=1.0, color=rgb_colors)

        # Добавляем прозрачные столбцы для отображения значений
        plt.bar(x, red_counts, color='red', alpha=0.3, label='Red')
        plt.bar(x, green_counts, color='green', alpha=0.3, label='Green')
        plt.bar(x, blue_counts, color='blue', alpha=0.3, label='Blue')

        # Настройка графика
        plt.title('Аддитивное смешение цветов точек в LAS-файле', pad=20)
        plt.xlabel('Значение цвета (0-255)')
        plt.ylabel('Количество точек')
        plt.legend()

        # Добавляем информацию о самых частых цветах
        max_combined = red_counts + green_counts + blue_counts
        top5_indices = np.argsort(-max_combined)[:5]

        for i in top5_indices:
            r, g, b = red_counts[i], green_counts[i], blue_counts[i]
            plt.text(i, max_combined[i],
                     f'R:{r}\nG:{g}\nB:{b}',
                     ha='center', va='bottom',
                     bbox=dict(facecolor='white', alpha=0.8))

        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlim(0, bins)
        plt.tight_layout()
        plt.show()

        return {
            'red': red_counts,
            'green': green_counts,
            'blue': blue_counts,
            'rgb_colors': rgb_colors
        }

    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")
        return None


# Пример использования:
hist_data = plot_color_histogram_stacked("pp20cutbuf5NORM-norm.las")