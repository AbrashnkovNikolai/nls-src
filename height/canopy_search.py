import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from clustering import process_segment_reclustering
from tree_detection.rasterization import build_chm
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def canopy_search(chm, chm_bounds, tree_markers, centers_dict, chm_resolution=0.5):
    """
    chm: растровый слой CHM из build_chm_with_stratification
    chm_bounds: кортеж (min_x, min_y, max_x, max_y) из build_chm_with_stratification
    tree_markers: список координат (x, y) стволов в реальных координатах
    centers_dict: словарь с центрами деревьев {seg_id: (x, y, radius)}
    chm_resolution: разрешение CHM (0.5 по умолчанию)
    """
    min_x, min_y, max_x, max_y = chm_bounds

    # 1. Конвертируем реальные координаты в пиксельные координаты CHM
    tree_markers_pixels = []
    for x, y in tree_markers:
        px = int((x - min_x) / chm_resolution)
        py = int((y - min_y) / chm_resolution)
        # Проверяем, чтобы координаты не выходили за границы CHM
        px = max(0, min(px, chm.shape[1] - 1))
        py = max(0, min(py, chm.shape[0] - 1))
        tree_markers_pixels.append((px, py))

    # 2. Создаем маркерную матрицу
    markers = np.zeros_like(chm, dtype=np.int32)
    seg_ids = list(centers_dict.keys())

    for i, (px, py) in enumerate(tree_markers_pixels, start=1):
        markers[py, px] = i

    # 3. Применяем Watershed
    chm_inverted = -chm
    # Используем маску для исключения низкой растительности (порог можно настроить)
    vegetation_mask = chm > 2.0
    segmented_crowns = watershed(chm_inverted, markers, mask=vegetation_mask)

    # 4. Для каждого сегмента находим максимальную высоту и создаем результат
    results = {}

    for marker_id in np.unique(segmented_crowns)[1:]:  # игнорируем фон (0)
        crown_mask = (segmented_crowns == marker_id)
        height_in_crown = chm[crown_mask]

        if len(height_in_crown) > 0:
            max_height = np.max(height_in_crown)

            # Сопоставляем marker_id с seg_id
            marker_index = marker_id - 1
            if marker_index < len(seg_ids):
                seg_id = seg_ids[marker_index]
                center_x, center_y, radius = centers_dict[seg_id]

                results[seg_id] = {
                    'x': center_x,
                    'y': center_y,
                    'z': max_height,
                    'd': radius,
                    's': 0,
                    'h': max_height,
                }

    print(f"Успешно обработано {len(results)} деревьев из {len(seg_ids)}")
    return results


def improved_canopy_search(chm, chm_bounds, tree_markers, centers_dict,
                           chm_resolution=0.5, min_height_threshold=1.5,
                           crown_smoothing=True, watershed_compactness=0.01):
    """
    Улучшенная версия с настраиваемыми параметрами
    """
    min_x, min_y, max_x, max_y = chm_bounds

    # Параметр 1: Минимальная высота растительности
    vegetation_mask = chm > min_height_threshold  # Настройка!

    # Параметр 2: Сглаживание CHM для уменьшения шума
    if crown_smoothing:
        from scipy import ndimage
        chm_smooth = ndimage.gaussian_filter(chm, sigma=0.85)
    else:
        chm_smooth = chm

    # Конвертация координат
    tree_markers_pixels = []
    for x, y in tree_markers:
        px = int((x - min_x) / chm_resolution)
        py = int((y - min_y) / chm_resolution)
        px = max(0, min(px, chm.shape[1] - 1))
        py = max(0, min(py, chm.shape[0] - 1))
        tree_markers_pixels.append((px, py))

    # Создание маркеров
    markers = np.zeros_like(chm_smooth, dtype=np.int32)
    seg_ids = list(centers_dict.keys())

    for i, (px, py) in enumerate(tree_markers_pixels, start=1):
        markers[py, px] = i

    # Параметр 3: Watershed
    chm_inverted = -chm_smooth
    segmented_crowns = watershed(chm_inverted, markers,
                                 mask=vegetation_mask,
                                 compactness=watershed_compactness)

    # Улучшенное определение высоты: используем 95-й перцентиль вместо максимума
    results = {}
    for marker_id in np.unique(segmented_crowns)[1:]:
        crown_mask = (segmented_crowns == marker_id)
        height_in_crown = chm_smooth[crown_mask]

        if len(height_in_crown) > 10:  # Минимальное количество точек в кроне
            # Параметр 4: Используем 95-й перцентиль вместо максимума
            max_height = np.percentile(height_in_crown, 99)

            marker_index = marker_id - 1
            if marker_index < len(seg_ids):
                seg_id = seg_ids[marker_index]
                center_x, center_y, radius = centers_dict[seg_id]

                results[seg_id] = {
                    'x': center_x, 'y': center_y,
                    'z': max_height, 'd': radius,
                    's': 0, 'h': max_height,
                }

    return results


def height_pipeline(segments, centers_dict, all_points, output_path):
    print(f"Запуск пайплайна с фильтрацией интервалов для {len(segments)} деревьев...")

    points_array = np.array(all_points, dtype=np.float32)

    # Параллельная обработка с вашей функцией фильтрации
    results = Parallel(n_jobs=1)(
        delayed(process_segment_reclustering)(
            seg_id, segment_points, centers_dict
        )
        for seg_id, segment_points in segments.items()
    )

    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_path, sep=';', index=False)

        print(f"✅ Успешно обработано {len(final_df)} деревьев")
        print(f"📊 Статистика высот:")
        print(f"   Мин. высота: {final_df['h'].min():.2f} м")
        print(f"   Макс. высота: {final_df['h'].max():.2f} м")
        print(f"   Сред. высота: {final_df['h'].mean():.2f} м")

        return final_df
    else:
        print("❌ Не удалось обработать деревья")
        return pd.DataFrame()