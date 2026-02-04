import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import KDTree
import numpy as np
from numpy.typing import NDArray


def find_height_in_segment(seg_id, segment_points, centers_dict):
    """Находит высоту дерева с предварительной фильтрацией точек по разрывам"""

    # Получаем центр дерева
    center = centers_dict[seg_id]
    x, y, radius = center

    # 1. ФИЛЬТРАЦИЯ: удаляем нижние точки при больших разрывах в высоте
    if len(segment_points) > 1:
        z = segment_points[:, 2]
        sorted_idx = np.argsort(z)[::-1]  # индексы от самых высоких к низким
        sorted_z = z[sorted_idx]

        # Ищем первый разрыв ≥ 0.5 метра
        for i in range(1, len(sorted_z)):
            if sorted_z[i - 1] - sorted_z[i] >= 0.5:
                # Оставляем только точки выше разрыва
                segment_points = segment_points[sorted_idx[:i]]
                break

    # 2. РАСЧЕТ ВЫСОТЫ
    if len(segment_points) == 0:
        tree_height = 0
    else:
        points_z = segment_points[:, 2]

        # Если точек меньше 2, просто берем максимальную высоту
        if len(points_z) < 2:
            tree_height = np.max(points_z)
        else:
            sorted_z = np.sort(points_z)

            # Определяем количество бинов для гистограммы
            # Минимум 2 бина, максимум 20
            num_bins = min(20, max(2, len(points_z) // 5))

            # Гистограмма для нахождения кроны
            hist, bin_edges = np.histogram(sorted_z, bins=num_bins)
            main_crown_bin = np.argmax(hist)
            crown_top = bin_edges[main_crown_bin + 1]

            # Верхняя граница кроны
            top_height = crown_top
            for i in range(main_crown_bin + 1, len(hist)):
                if hist[i] < max(hist) * 0.05:
                    top_height = bin_edges[i]
                    break
            else:
                top_height = np.max(points_z)

            # Проверка на разрыв с другим деревом
            tree_height = top_height
            for i in range(len(sorted_z) - 1):
                if sorted_z[i] > crown_top and (sorted_z[i + 1] - sorted_z[i]) > 1.5:
                    tree_height = sorted_z[i]
                    break

    return {
        'x': center[0],
        'y': center[1],
        'z': tree_height,
        'd': center[2],
        's': 0,
        'h': tree_height,
    }
def build_chm_with_stratification(
        points: np.ndarray,
        pixel_size: float,
        height_thresholds: list[float] = [1.0, 5.0, 10.0, 15.0],
        chunk_size: int = 10_000_000
) -> tuple:
    """
    Построение CHM с учетом стратификации (слоев) растительности.
    """
    points_array = np.array(points)

    # Определяем границы области
    x_coords, y_coords = points_array[:, 0], points_array[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    width = int(np.ceil((max_x - min_x) / pixel_size))
    height = int(np.ceil((max_y - min_y) / pixel_size))

    print(f"Построение многослойного CHM: {width} x {height} пикселей")

    # Инициализируем итоговый CHM
    final_chm = np.zeros((height, width), dtype=np.float32)
    processed_mask = np.zeros((height, width), dtype=bool)

    # Обрабатываем слои от высоких к низким
    height_thresholds_sorted = sorted(height_thresholds, reverse=True)

    for i, threshold in enumerate(height_thresholds_sorted):
        print(f"Обработка слоя высот > {threshold} м")

        # Фильтруем точки текущего слоя
        if i == 0:
            # Самый высокий слой - все точки выше порога
            layer_mask = points_array[:, 2] >= threshold
        else:
            # Промежуточные слои - точки между текущим и предыдущим порогом
            prev_threshold = height_thresholds_sorted[i - 1]
            layer_mask = (points_array[:, 2] >= threshold) & (points_array[:, 2] < prev_threshold)

        layer_points = points_array[layer_mask]

        if len(layer_points) == 0:
            continue

        # Строим CHM для текущего слоя
        layer_chm = _build_layer_chm(
            layer_points, min_x, min_y, width, height, pixel_size, chunk_size
        )

        # Заполняем только еще не обработанные пиксели
        new_pixels_mask = (layer_chm > 0) & ~processed_mask
        final_chm[new_pixels_mask] = layer_chm[new_pixels_mask]
        processed_mask[new_pixels_mask] = True

    # Заполняем оставшиеся пропуски нижним слоем (< минимального порога)
    low_layer_mask = points_array[:, 2] < height_thresholds_sorted[-1]
    low_points = points_array[low_layer_mask]

    if len(low_points) > 0 and not np.all(processed_mask):
        low_chm = _build_layer_chm(
            low_points, min_x, min_y, width, height, pixel_size, chunk_size
        )
        remaining_mask = ~processed_mask & (low_chm > 0)
        final_chm[remaining_mask] = low_chm[remaining_mask]

    return final_chm, min_x, min_y, max_x, max_y


def _build_layer_chm(points, min_x, min_y, width, height, pixel_size, chunk_size):
    """Вспомогательная функция для построения CHM слоя"""
    # Создаем пустой CHM для слоя
    layer_chm = np.zeros((height, width), dtype=np.float32)

    # Рассчитываем индексы пикселей для каждой точки
    x_indices = ((points[:, 0] - min_x) / pixel_size).astype(int)
    y_indices = ((points[:, 1] - min_y) / pixel_size).astype(int)

    # Убеждаемся, что индексы в пределах массива
    valid_mask = (x_indices >= 0) & (x_indices < width) & (y_indices >= 0) & (y_indices < height)
    x_indices = x_indices[valid_mask]
    y_indices = y_indices[valid_mask]
    heights = points[valid_mask, 2]

    # Для каждого пикселя берем максимальную высоту
    for i in range(len(x_indices)):
        x, y = x_indices[i], y_indices[i]
        if heights[i] > layer_chm[y, x]:
            layer_chm[y, x] = heights[i]

    return layer_chm


def filter_segmentation_interval(
        points: np.ndarray,
        s: np.ndarray,
        max_interval: float,
        indent: int = 2
) -> np.ndarray:

    if len(points) == 0:
        return s

    # Берем только Z координаты точек СЕГМЕНТА
    z_coords = points[:, 2]

    k = 0
    for i in range(0, np.max(s) + 1):
        indexes = np.where(s == i)[0]
        if len(indexes) == 0:
            continue

        z = z_coords[indexes]
        if len(z) < 2:
            continue

        # Сортируем высоты по убыванию
        sorted_indices = np.argsort(z)[::-1]
        sorted_z = z[sorted_indices]
        sorted_original_indices = indexes[sorted_indices]

        n = len(sorted_z)
        j = 1
        # Ищем первый разрыв превышающий max_interval
        while j < n and (sorted_z[j - 1] - sorted_z[j]) < max_interval:
            j += 1

        # Если нашли разрыв, помечаем все точки выше разрыва как -1
        if j < n:
            for idx in sorted_original_indices[:j]:  # точки ВЫШЕ разрыва
                s[idx] = -1
                k += 1

    print(f'{" " * indent}Точек в сегменте: {len(points)}, осталось после фильтрации: {len(points) - k}')
    return s

def fast_height_pipeline(segments, centers_dict, all_points, output):
    """
    пример распараллеливания функции, в случае долго обработки
    """
    print(f"Обработка {len(segments)} сегментов...")

    # Преобразуем в numpy array один раз
    points_array = np.array(all_points, dtype=np.float32)

    # Параллельная обработка
    results = Parallel(n_jobs=2)(
        delayed(find_height_in_segment)(
            seg_id, segment_points, centers_dict, points_array
        )
        for seg_id, segment_points in segments.items()
    )
    return results