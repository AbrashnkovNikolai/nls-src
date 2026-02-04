from shapely import contains

from other_tools.hight_profile_functions import *
from other_tools.cluseterToYOLO import points_to_image

import matplotlib.path as mpath

las_path = '../pp20cutbuf5NORM-norm.las'
#las_points =  get_sliced_points(las_path,-1000,1000)
test = [
    (357430.910, 6758474.922),  # left
    (357462.474, 6758520.326),  # top
    (357550.327, 6758456.846),  # right
    (357512.8682, 6758413.7518)  # bot
]
from imports import *
from las_functions import las_to_points
from other_tools.hight_profile_functions import *
from image_functions import *

def create_slice_segments(polygon_border, slice_width=3.0):
    """
    Создает сегменты срезов для полигона

    Parameters:
    polygon_border: list of tuples - границы полигона в порядке [left, top, right, bottom]
    slice_width: float - ширина среза в метрах

    Returns:
    list - список сегментов (каждый сегмент - список из 4 вершин прямоугольника)
    """
    left, top, right, bottom = polygon_border

    # Создаем точки на верхней и нижней границах
    upper_points = generate_points_on_line(left, top, step=slice_width)
    lower_points = generate_points_on_line(bottom, right, step=slice_width)

    # Выравниваем количество точек (берем минимальное количество)
    min_points = min(len(upper_points), len(lower_points))
    upper_points = upper_points[:min_points]
    lower_points = lower_points[:min_points]

    # Создаем сегменты-прямоугольники
    segments = []
    for i in range(min_points):
        segment = build_rectangle_from_points(upper_points[i], lower_points[i], slice_width / 2)
        segments.append(segment)

    return segments


def get_points_in_segment(segment_polygon, las_points):
    """
    Фильтрует точки LAS, попадающие в сегмент

    Parameters:
    segment_polygon: list of tuples - вершины полигона сегмента
    las_points: list - точки из LAS файла

    Returns:
    list - точки, попадающие в сегмент
    """
    from shapely.geometry import Point, Polygon

    segment_poly = Polygon(segment_polygon)
    points_in_segment = []

    for point in las_points:
        if len(point) >= 2:  # Проверяем, что есть хотя бы X,Y координаты
            point_obj = Point(point[0], point[1])
            if segment_poly.contains(point_obj):
                points_in_segment.append(point)

    return points_in_segment



def get_points_in_segment_vectorized(segment_polygon, las_points):
    """
    Векторизованная фильтрация с matplotlib.path
    """


    # Конвертируем в numpy array
    points_array = np.array(las_points)

    # Создаем path из полигона
    polygon_path = mpath.Path(segment_polygon)

    # Векторизованная проверка принадлежности
    mask = polygon_path.contains_points(points_array[:, :2])

    return points_array[mask].tolist()

def get_sliced_points_pipeline(las_path, polygon_border, slice_width=3.0, z_min=-1000, z_max=1000):
    """
    Основной пайплайн для получения срезов точек

    Parameters:
    las_path: str - путь к LAS файлу
    polygon_border: list of tuples - границы полигона
    slice_width: float - ширина среза в метрах
    z_min, z_max: float - диапазон высот для фильтрации

    Returns:
    list - список срезов, где каждый срез содержит точки этого сегмента
    """
    # 1. Загружаем и фильтруем точки LAS
    print("Загрузка точек LAS...")

    las_points = get_sliced_points(las_path, z_min, z_max)
    print(f"Загружено точек: {len(las_points)}")

    # 2. Создаем сегменты срезов
    print("Создание сегментов срезов...")
    segments = create_slice_segments(polygon_border, slice_width)
    print(f"Создано сегментов: {len(segments)}")

    # 3. Для каждого сегмента получаем точки
    print("Фильтрация точек по сегментам...")
    sliced_results = []

    for i, segment in enumerate(segments):
        segment_points = get_points_in_segment_vectorized(segment, las_points)
        image, _ = points_to_image(segment_points,point_size=2)
        cv2.imwrite(f'proections-xy/pp57xz-vls/{i}.png', image)
        sliced_results.append({
            'segment_id': i,
            'segment_polygon': segment,
            'points': segment_points,
            'points_count': len(segment_points)
        })
        print(f"Сегмент {i}: {len(segment_points)} точек")


    return sliced_results



# Пример использования
if __name__ == "__main__":
    # Определяем полигон
    test_border = [
        (357430.910, 6758474.922),  # left
        (357462.474, 6758520.326),  # top
        (357550.327, 6758456.846),  # right
        (357512.8682, 6758413.7518)  # bot
    ]
    pp57 = [(539482.873,6811851.002), #left
            (539519.3211,6811961.2148),
            (539586.3401,6811939.0043),
            (539549.7717,6811828.7270)
            ]
    # Запускаем пайплайн
    results = get_sliced_points_pipeline(
        las_path='F:/downloads/pp57-vls/pp57-vls-cut-120.las',
        polygon_border=pp57,
        slice_width=3,  #
        z_min=-1000,
        z_max=1000
    )

    # Выводим статистику
    total_points = sum(slice_data['points_count'] for slice_data in results)
    print(f"\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Всего сегментов: {len(results)}")
    print(f"Всего точек в сегментах: {total_points}")

    for slice_data in results:
        print(f"Сегмент {slice_data['segment_id']}: {slice_data['points_count']} точек")