from time import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ultralytics import YOLO

from functions.image_functions import process_point_cloud
from functions.las_functions import get_sliced_points
from height.canopy_search import canopy_search, improved_canopy_search, height_pipeline
from diam.clustering import *
from functions.filters_and_metrics import *
from params_and_parser import parse_parameters,config


from height.rasterization import build_chm_with_stratification
from functions.segmentation import segment_points_by_cylinders

from functions.tree_height_functions import (
    build_chm_with_stratification, find_height_in_segment, filter_segmentation_interval
)

def run():
    parameters = parse_parameters()
    start = time()
    BORDER = config['border_polygon']
    try:
        results = []
        existing_circles = None

        # Обработка по слоям
        min_h = config['LAYER'][0]
        max_h = config['LAYER'][1]
        print(f"\nОбработка слоя {min_h}-{max_h} м:")
        points = get_sliced_points(min_h, max_h)
        layer_result = process_layer(points[:, :2], existing_circles)

        if layer_result is not None:
            results.append(layer_result)
        else:
            print(f'layer_result is None: {layer_result is None}')

        # for min_h, max_h in LAYER:
        # Сохранение результатов
        if results:

            df_circles = pd.DataFrame(pd.concat(results, ignore_index=True))
            # df_circles.to_csv(output, sep =";")
            print(f"Сохранено {len(df_circles)} деревьев")

            x = df_circles['x'].values
            y = df_circles['y'].values
            r = df_circles['d'].values
            cirk_centers_2d = list(zip(x, y, r))
            #insides = filter_by_border(cirk_centers_2d, BORDER)
            # [(x1, y1, r1), (x2, y2, r2), ...]

            # Создаем и сохраняем сегменты
            #all_points = get_all_points()
            #segments = segment_points_by_frustum_cones(all_points, insides, height=parametrs['min_height'])
            if config['create_segmented_las']:
                segments_path = config['segments_las_path']
            #    save_segments_to_las(segments, segments_path)

            #df = pd.DataFrame(calculate_segment_heights_from_dict(segments, insides))
            #df.to_csv(output, sep=';')
    finally:
        print(f"Время выполнения: {time() - start:.2f} секунд")


def run_yolo1(las_file_path, output,model_path):

    #BORDER = parametrs['border_polygon']

    yolo_model = YOLO(model_path)
    all_points = get_sliced_points(las_file_path, 0.5, 50)
    #height_points = raster_height_search(all_points)
    sliced_points = get_sliced_points(las_file_path, 1.3, 1.5)
    cirk_results = []
    labels = clusterize_points(sliced_points)
    for label in set(labels):
        # Получаем точки кластера
        cluster_points = sliced_points[labels == label]
        # Обрабатываем кластер
        circle_coords, image, _ = process_point_cloud(cluster_points, yolo_model)
        # в circle_coords лежит [(x,y,r),...]
        if circle_coords:
            d = (circle_coords[2] * 2) - config['circle_thickness']
            cirk_results.append({
                'x': circle_coords[0],
                'y': circle_coords[1],
                'z': 1.3,
                'd': d,
                's': 0,
                'h': 0,
                # 'geometry': Point(center[0], center[1]).buffer(radius)
            })

    df = pd.DataFrame(cirk_results)
    x = df['x'].values
    y = df['y'].values
    r = df['d'].values

    # [(x1, y1, r1), (x2, y2, r2), ...]
    cirk_centers_3d = list(zip(x, y, r))
    #pairs = find_closest_pairs(height_points,cirk_centers_3d)

    #results = []
    '''for pair in pairs:
        #pair[1] точка ствола в паре
        #pair[0] высотная точка в паре
        results.append({
                'x': pair[1][0],
                'y': pair[1][1],
                'z': pair[0][2],
                'd': pair[1][2],
                's': 0,
                'h': pair[0][2],
        }
        )'''


    results = []
    # insides = det.filter_by_border(cirk_centers_2d, BORDER)
    segments, centers_dict = segment_points_by_cylinders(all_points, cirk_centers_3d)


    def process_single_segment(seg_id, segment_points, centers_dict):
        center = centers_dict[seg_id]
        x, y, radius = center
        lmf = find_highest_near_axis_priority_height(segment_points, (x, y, radius))
        return {
            'x': center[0],
            'y': center[1],
            'z': lmf[1],
            'd': center[2],
            's': 0,
            'h': lmf[1],
        }

    # Строим CHM и получаем границы
    chm, min_x, min_y, max_x, max_y = build_chm_with_stratification(
        all_points,
        pixel_size=1.0,  # Увеличили разрешение для лучшей работы
        height_thresholds=[1.0, 3.0, 8.0, 15.0]
    )

    chm_bounds = (min_x, min_y, max_x, max_y)



    # Создаем финальный DataFrame
    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(output, sep=';')
        print(f"Сохранено {len(final_df)} деревьев с методом разрыва высот")
    else:
        print("Не удалось обработать деревья")
        # Fallback - создаем пустой DataFrame с правильной структурой
        final_df = pd.DataFrame(columns=['x', 'y', 'z', 'd', 's', 'h'])
        final_df.to_csv(output, sep=';')
    # Построение CHM с улучшенными параметрами


    # Создаем и сохраняем результаты
    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(output, sep=';')
        print(f"Успешно обработано {len(final_df)} деревьев")
        return final_df
    else:
        print("Не удалось обработать деревья")
        return pd.DataFrame()

def run_yolo(las_file_path, output, model_path):
    """
    Основной пайплайн обработки LiDAR данных с поиском деревьев
    """
    # Загрузка модели YOLO
    yolo_model = YOLO(model_path)

    # Получение точек
    all_points = get_sliced_points(las_file_path, 1.3, 50)
    sliced_points = get_sliced_points(las_file_path, 1.3, 1.5)

    # Кластеризация и обработка YOLO
    cirk_results = []
    labels = clusterize_points(sliced_points)

    for label in set(labels):
        cluster_points = sliced_points[labels == label]
        circle_coords, image, _ = process_point_cloud(cluster_points, yolo_model)

        if circle_coords:
            d = (circle_coords[2] * 2) - config['circle_thickness']
            cirk_results.append({
                'x': circle_coords[0],
                'y': circle_coords[1],
                'z': 1.3,
                'd': d,
                's': 0,
                'h': 0,
            })

    # Создаем центры деревьев
    df = pd.DataFrame(cirk_results)
    x = df['x'].values
    y = df['y'].values
    d = df['d'].values
    cirk_centers_3d = list(zip(x, y, d))

    # Сегментация точек по цилиндрам
    segments, centers_dict = segment_points_by_cylinders(all_points, cirk_centers_3d,buffer=1.5)

    print(f"Загружено {len(segments)} сегментов")

    # Для каждого сегмента определяем высоту дерева
    results = []
    for seg_id, segment_points in segments.items():
        height_data = find_height_in_segment(seg_id, segment_points, centers_dict)
        results.append(height_data)

    # Создаем финальный DataFrame с высотами
    final_df = pd.DataFrame(results)

    # Сохраняем результаты если нужно
    if output:
        final_df.to_csv(output, sep=';', index=False)
        print(f"Результаты сохранены в {output}")

    return final_df

