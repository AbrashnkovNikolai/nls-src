import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

import rasterio
from rasterio.transform import from_origin


def save_raster_to_tiff(
        raster: np.ndarray,
        output_path: str,
        min_x: float,
        min_y: float,
        pixel_size: float,
        crs: str = "EPSG:32640"
):
    """
    Сохраняет растр в файл TIFF с правильной ориентацией.
    """
    height, width = raster.shape

    raster_data = np.flipud(raster.copy())
    raster_data = raster_data.astype(float)

    transform = from_origin(min_x, min_y + height * pixel_size, pixel_size, pixel_size)

    if np.any(np.isnan(raster_data)):
        raster_data = np.nan_to_num(raster_data, nan=-9999.0)
        nodata = -9999.0
    else:
        nodata = None

    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=raster_data.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata
    ) as dst:
        dst.write(raster_data, 1)

    return None


def visualize_raster_white_background(
        raster,
        points,
        min_x,
        min_y,
        title: str = "Растр с белым фоном",
        cmap: str = "viridis",
        point_color: str = "red",
        point_alpha: float = 0.3,
        point_size: float = 10,
):
    """
    Визуализирует растр с белым фоном для пустых областей и отображает точки поверх растра.
    """
    plt.figure(figsize=(10, 8))

    cmap_with_white = plt.cm.get_cmap(cmap).copy()
    cmap_with_white.set_bad('white', alpha=1.0)
    im = plt.imshow(raster, cmap=cmap_with_white, origin='lower')

    if points is not None:
        points_array = np.array(points)
        if points_array.size > 0:
            height, width = raster.shape
            x_coords = points_array[:, 0] - min_x
            y_coords = points_array[:, 1] - min_y

            x_scale = width / (np.max(x_coords) if np.max(x_coords) > 0 else 1)
            y_scale = height / (np.max(y_coords) if np.max(y_coords) > 0 else 1)

            x_scaled = x_coords * x_scale
            y_scaled = y_coords * y_scale

            plt.scatter(x_scaled, y_scaled, c=point_color, alpha=point_alpha, s=point_size, marker='^')

    plt.colorbar(im, label='Степень заполненности')
    plt.title(title)
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.tight_layout()
    plt.show()


def build_chm_with_stratification(
        points: np.ndarray,
        pixel_size: float,
        height_thresholds: list[float] = [5.0, 10.0, 15.0,20.0,25.0,30.0,35.0],
        chunk_size: int = 10_000_000
) -> Tuple[np.ndarray, float, float, float, float]:
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
    """Построение CHM для отдельного слоя"""
    chm = np.zeros((height, width), dtype=np.float32)

    for start_idx in range(0, len(points), chunk_size):
        end_idx = min(start_idx + chunk_size, len(points))
        chunk = points[start_idx:end_idx]

        x_idx = ((chunk[:, 0] - min_x) / pixel_size).astype(int)
        y_idx = ((chunk[:, 1] - min_y) / pixel_size).astype(int)
        z_vals = chunk[:, 2]

        valid_mask = (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)
        valid_y, valid_x, valid_z = y_idx[valid_mask], x_idx[valid_mask], z_vals[valid_mask]

        if len(valid_z) > 0:
            np.maximum.at(chm, (valid_y, valid_x), valid_z)

    return chm


def build_chm(
        points: np.ndarray,
        pixel_size: float,
        fill_gaps: bool = True,
        max_gap_distance: float = 3.0,
        chunk_size: int = 10_000_000
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Построение цифровой модели высот крон (CHM) из нормализованных точек.

    Parameters:
    -----------
    points : np.ndarray
        Массив точек [x, y, z] с нормализованными высотами (z - высота над землей)
    pixel_size : float
        Размер пикселя в метрах
    fill_gaps : bool
        Заполнять ли пропуски в данных
    max_gap_distance : float
        Максимальное расстояние для заполнения пропусков (в метрах)
    chunk_size : int
        Размер чанка для обработки больших datasets

    Returns:
    --------
    tuple : (chm_raster, min_x, min_y, max_x, max_y)
        Растр CHM и границы области
    """
    points_array = np.array(points)
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]

    # Определяем границы области
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # Вычисляем размеры растра
    width = int(np.ceil((max_x - min_x) / pixel_size))
    height = int(np.ceil((max_y - min_y) / pixel_size))

    print(f"Построение CHM: {width} x {height} пикселей, разрешение: {pixel_size} м")

    # Инициализируем растр с no-data значениями
    chm_raster = np.full((height, width), -9999.0, dtype=np.float32)

    # Обрабатываем точки по чанкам
    n_points = len(points)
    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)
        chunk_points = points_array[start_idx:end_idx]

        print(f"Обработка чанка {start_idx}-{end_idx} из {n_points}")

        # Векторизованное вычисление индексов
        x_indices = ((chunk_points[:, 0] - min_x) / pixel_size).astype(int)
        y_indices = ((chunk_points[:, 1] - min_y) / pixel_size).astype(int)
        z_values = chunk_points[:, 2]

        # Фильтрация валидных индексов
        valid_mask = (x_indices >= 0) & (x_indices < width) & \
                     (y_indices >= 0) & (y_indices < height) & \
                     (z_values >= 0)  # Игнорируем отрицательные высоты

        valid_y = y_indices[valid_mask]
        valid_x = x_indices[valid_mask]
        valid_z = z_values[valid_mask]

        # Обновляем максимальные высоты для каждого пикселя
        if len(valid_z) > 0:
            # Создаем временный растр для текущего чанка
            chunk_raster = np.full((height, width), -9999.0, dtype=np.float32)

            # Используем maximum.at для аккумуляции максимальных значений
            np.maximum.at(chunk_raster, (valid_y, valid_x), valid_z)

            # Объединяем с основным растром
            update_mask = chunk_raster > chm_raster
            chm_raster[update_mask] = chunk_raster[update_mask]

    # Заполняем пропуски если требуется
    if fill_gaps:
        chm_raster = _fill_chm_gaps(chm_raster, pixel_size, max_gap_distance)

    # Применяем легкое сглаживание для уменьшения шума
    chm_raster = _smooth_chm(chm_raster)

    # Заменяем оставшиеся no-data значения на 0
    chm_raster[chm_raster == -9999.0] = 0.0

    print(f"CHM построен. Диапазон высот: {np.min(chm_raster):.2f} - {np.max(chm_raster):.2f} м")

    return chm_raster, min_x, min_y, max_x, max_y


def _fill_chm_gaps(
        chm: np.ndarray,
        pixel_size: float,
        max_gap_distance: float
) -> np.ndarray:
    """
    Заполнение пропусков в CHM.
    """
    from scipy.ndimage import distance_transform_edt

    # Создаем маску валидных данных
    valid_mask = chm > 0
    invalid_mask = ~valid_mask

    # Если нет пропусков, возвращаем исходный растр
    if not np.any(invalid_mask):
        return chm

    # Вычисляем максимальное расстояние для заполнения в пикселях
    max_gap_pixels = int(max_gap_distance / pixel_size)

    # Заполняем только небольшие пропуски
    if max_gap_pixels > 0:
        # Находим расстояния до ближайших валидных пикселей
        distances, indices = distance_transform_edt(
            invalid_mask,
            return_indices=True,
            return_distances=True
        )

        # Заполняем пропуски в пределах максимального расстояния
        fill_mask = invalid_mask & (distances <= max_gap_pixels)

        if np.any(fill_mask):
            # Берем значения из ближайших валидных пикселей
            chm_filled = chm.copy()
            chm_filled[fill_mask] = chm[tuple(indices[:, fill_mask])]
            return chm_filled

    return chm


def _smooth_chm(chm: np.ndarray, sigma: float = 0.7) -> np.ndarray:
    """
    Легкое сглаживание CHM для уменьшения шума.
    """
    from scipy.ndimage import gaussian_filter

    # Сглаживаем только валидные данные (высоты > 0)
    valid_mask = chm > 0

    if np.sum(valid_mask) > 0:
        smoothed_chm = chm.copy()
        smoothed_chm[valid_mask] = gaussian_filter(
            chm[valid_mask],
            sigma=sigma,
            mode='nearest'
        )
        return smoothed_chm

    return chm


def build_raster_chunked_discrete(
        points: np.ndarray,
        z_step: float,
        pixel_size: float,
        r: int = 1,
        chunk_size: int = 10_000_000
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Построение растра с чанкованием для больших наборов данных.
    Степень заполненности пикселя определяется дискретно по высоте с заданным шагом.

    Args:
        points: Список точек (x, y, z)
        z_step: Шаг дискретизации по высоте
        pixel_size: Размер пикселя
        chunk_size: Размер чанка для обработки
        r: Радиус влияния точек (количество соседних пикселей)
    """

    points_array = np.array(points)
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    z_coords = points_array[:, 2]

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    min_z, max_z = np.min(z_coords), np.max(z_coords)

    width = int(np.ceil((max_x - min_x) / pixel_size))
    height = int(np.ceil((max_y - min_y) / pixel_size))

    print(f"Размер растра: {width} x {height} пикселей")
    print(f"Диапазон X: {min_x:.2f} - {max_x:.2f}")
    print(f"Диапазон Y: {min_y:.2f} - {max_y:.2f}")
    print(f"Диапазон Z: {min_z:.2f} - {max_z:.2f}")
    print(f"Шаг по Z: {z_step:.2f}")
    print(f"Радиус влияния: {r} пикселей")

    # Количество слоев по Z
    num_z_layers = max(1, int(np.ceil((max_z - min_z) / z_step)))
    print(f"Количество слоев по Z: {num_z_layers}")

    # Инициализируем 3D массив для хранения заполненности по слоям
    # Форма: (height, width, num_z_layers)
    layer_fill = np.zeros((height, width, num_z_layers), dtype=bool)

    n_points = len(points)
    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)
        chunk_points = points_array[start_idx:end_idx]

        # print(f"Обработка чанка {start_idx}-{end_idx} из {n_points}")

        for point in chunk_points:
            x, y, z = point

            # Определяем индекс пикселя
            x_idx = int((x - min_x) / pixel_size)
            y_idx = int((y - min_y) / pixel_size)

            if 0 <= x_idx < width and 0 <= y_idx < height:
                # Определяем слой по Z
                z_layer = int((z - min_z) / z_step)
                z_layer = max(0, min(num_z_layers - 1, z_layer))

                # Отмечаем заполненность в центральном пикселе и соседних
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        neighbor_y = y_idx + dy
                        neighbor_x = x_idx + dx

                        # Проверяем границы растра
                        if (0 <= neighbor_x < width and
                                0 <= neighbor_y < height):
                            layer_fill[neighbor_y, neighbor_x, z_layer] = True

    # Вычисляем степень заполненности для каждого пикселя
    # как отношение заполненных слоев к общему количеству слоев
    raster = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            filled_layers = np.sum(layer_fill[y, x, :])
            if filled_layers > 0:
                raster[y, x] = filled_layers / num_z_layers
                if raster[y, x] < 0.1:
                    raster[y, x] = 0

    return raster, min_x, min_y, max_x, max_y


def build_max_z_raster(
        points: np.ndarray,
        pixel_size: float,
        chunk_size: int = 10_000_000
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Построение растра максимальных высот.
    """
    points_array = np.array(points)
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    width = int(np.ceil((max_x - min_x) / pixel_size))
    height = int(np.ceil((max_y - min_y) / pixel_size))

    print(f"Размер растра максимальных высот: {width} x {height} пикселей")

    max_z_raster = np.full((height, width), -1, dtype=np.float32)

    n_points = len(points)
    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)
        chunk_points = points_array[start_idx:end_idx]

        print(f"Обработка чанка {start_idx}-{end_idx} из {n_points}")

        # Векторизованное вычисление индексов
        x_indices = ((chunk_points[:, 0] - min_x) / pixel_size).astype(int)
        y_indices = ((chunk_points[:, 1] - min_y) / pixel_size).astype(int)
        z_values = chunk_points[:, 2]

        # Фильтрация валидных индексов
        valid_mask = (x_indices >= 0) & (x_indices < width) & \
                     (y_indices >= 0) & (y_indices < height)

        valid_y = y_indices[valid_mask]
        valid_x = x_indices[valid_mask]
        valid_z = z_values[valid_mask]

        # Векторизованное обновление максимальных высот
        np.maximum.at(max_z_raster, (valid_y, valid_x), valid_z)

    return max_z_raster, min_x, min_y, max_x, max_y


def build_raster_chunked_point_count(
        points: np.ndarray,
        pixel_size: float,
        r: int = 1,
        chunk_size: int = 10_000_000,
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Построение растра на основе количества точек в пикселе.
    """
    points_array = np.array(points)
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    width = int(np.ceil((max_x - min_x) / pixel_size))
    height = int(np.ceil((max_y - min_y) / pixel_size))

    print(f"Размер растра: {width} x {height} пикселей")

    # Инициализируем массив для подсчета точек
    point_count = np.zeros((height, width), dtype=np.float32)

    n_points = len(points)
    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)
        chunk_points = points_array[start_idx:end_idx]

        print(f"Обработка чанка {start_idx}-{end_idx} из {n_points}")

        # Векторизованное вычисление индексов
        x_indices = ((chunk_points[:, 0] - min_x) / pixel_size).astype(int)
        y_indices = ((chunk_points[:, 1] - min_y) / pixel_size).astype(int)

        # Фильтрация валидных индексов
        valid_mask = (x_indices >= 0) & (x_indices < width) & \
                     (y_indices >= 0) & (y_indices < height)

        valid_x = x_indices[valid_mask]
        valid_y = y_indices[valid_mask]

        # Создаем расширенную гистограмму с учетом радиуса
        if r == 0:
            # Без радиуса - просто 2D гистограмма
            np.add.at(point_count, (valid_y, valid_x), 1)
        else:
            # С радиусом - добавляем точки для всех соседей
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    neighbor_x = valid_x + dx
                    neighbor_y = valid_y + dy

                    # Фильтруем по границам
                    neighbor_mask = (neighbor_x >= 0) & (neighbor_x < width) & \
                                    (neighbor_y >= 0) & (neighbor_y < height)

                    valid_nx = neighbor_x[neighbor_mask]
                    valid_ny = neighbor_y[neighbor_mask]

                    np.add.at(point_count, (valid_ny, valid_nx), 1)

    return point_count, min_x, min_y, max_x, max_y


def build_raster_height_variance(
        points,
        pixel_size,
        neighbor_radius=1,
        chunk_size=10_000_000
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Упрощенная версия с учетом соседних пикселей (все веса равны)
    """
    points_array = np.array(points)
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    width = int(np.ceil((max_x - min_x) / pixel_size))
    height = int(np.ceil((max_y - min_y) / pixel_size))

    height_dict = {}

    n_points = len(points)
    for start_idx in range(0, n_points, chunk_size):
        end_idx = min(start_idx + chunk_size, n_points)
        chunk_points = points_array[start_idx:end_idx]

        x_indices = ((chunk_points[:, 0] - min_x) / pixel_size).astype(int)
        y_indices = ((chunk_points[:, 1] - min_y) / pixel_size).astype(int)
        z_values = chunk_points[:, 2]

        valid_mask = (x_indices >= 0) & (x_indices < width) & \
                     (y_indices >= 0) & (y_indices < height)

        valid_x = x_indices[valid_mask]
        valid_y = y_indices[valid_mask]
        valid_z = z_values[valid_mask]

        # Собираем высоты с учетом соседей
        for x, y, z in zip(valid_x, valid_y, valid_z):
            for dy in range(-neighbor_radius, neighbor_radius + 1):
                for dx in range(-neighbor_radius, neighbor_radius + 1):
                    neighbor_y = y + dy
                    neighbor_x = x + dx

                    if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                        key = (neighbor_y, neighbor_x)
                        if key not in height_dict:
                            height_dict[key] = []
                        height_dict[key].append(z)

    variance_raster = np.zeros((height, width), dtype=np.float32)
    for (y, x), heights in height_dict.items():
        if len(heights) > 1:
            variance_raster[y, x] = np.var(heights)

    return variance_raster, min_x, min_y, max_x, max_y


def find_local_maxima_mask(
        raster: np.ndarray,
        neighborhood_size: int = 3,
        threshold: float = 0.1
) -> np.ndarray:
    """
    Поиск локальных максимумов в растре и возврат булевой матрицы.

    Args:
        raster: Растр с нормализованными значениями
        neighborhood_size: Размер окрестности для поиска максимумов (нечетное число)
        threshold: Пороговое значение для фильтрации слабых максимумов

    Returns:
        Булева матрица той же формы, что и raster, где True указывает на локальный максимум
    """

    height, width = raster.shape

    # Инициализируем булеву матрицу
    local_maxima_mask = np.zeros((height, width), dtype=bool)

    # Проверяем и корректируем размер окрестности
    if neighborhood_size % 2 == 0:
        neighborhood_size += 1
        print(f"Размер окрестности изменен на {neighborhood_size} (должен быть нечетным)")

    half_size = neighborhood_size // 2

    # Обрабатываем только внутреннюю область (избегаем границ)
    for y in range(half_size, height - half_size):
        for x in range(half_size, width - half_size):
            current_value = raster[y, x]

            # Пропускаем значения ниже порога
            if current_value < threshold:
                continue

            # Проверяем, является ли текущая точка локальным максимумом
            is_local_max = True
            neighborhood = raster[
                           y - half_size:y + half_size + 1,
                           x - half_size:x + half_size + 1
                           ]

            for ny in range(neighborhood_size):
                for nx in range(neighborhood_size):
                    if ny == half_size and nx == half_size:
                        continue  # Пропускаем центральную точку

                    if neighborhood[ny, nx] >= current_value:
                        is_local_max = False
                        break
                if not is_local_max:
                    break

            if is_local_max:
                local_maxima_mask[y, x] = True

    return local_maxima_mask


def maxima_mask_to_points(
        mask: np.ndarray,
        max_z_raster: np.ndarray,
        min_x: float,
        min_y: float,
        pixel_size: float
) -> np.ndarray:
    """
    Преобразует булеву маску локальных максимумов в массив точек с координатами.

    Args:
        mask: Булева матрица локальных максимумов (True, где максимум)
        max_z_raster: Растр с максимальными высотами в каждом пикселе
        min_x: Минимальная X-координата области
        min_y: Минимальная Y-координата области
        pixel_size: Размер пикселя в координатах

    Returns:
        Массив точек (x, y, z) в мировых координатах
    """

    mx = []
    my = []
    mz = []
    y_indices, x_indices = np.where(mask)

    pixel_size = np.float64(pixel_size)

    for y, x in zip(y_indices, x_indices):
        world_x = (x + 0.5) * pixel_size
        world_y = (y + 0.5) * pixel_size
        world_z = max_z_raster[y, x]

        mx.append(world_x)
        my.append(world_y)
        mz.append(world_z)

    mx = np.array(mx, dtype=np.float64)
    my = np.array(my, dtype=np.float64)
    mz = np.array(mz, dtype=np.float64)

    mx += min_x
    my += min_y

    return np.array((mx, my, mz)).T


def normalize(arr: np.array):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
