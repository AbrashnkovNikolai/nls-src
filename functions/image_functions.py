import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from shapely.geometry import Point
import geopandas as gpd
from pyproj import CRS


def pixel_to_coords(pixel_x, pixel_y, transform):
    return (np.array([pixel_x, pixel_y]) / transform['scale']) + transform['offset']


def filter_clusters_by_size(points, labels, min_points=30, max_points=None):
    """
    Фильтрует кластеры по количеству точек в них.

    Параметры:
    - points: массив точек формата (N, 2) или (N, 3)
    - labels: массив меток кластеров (N,)
    - min_points: минимальное количество точек в кластере (включительно)
    - max_points: максимальное количество точек (None - без ограничения)

    Возвращает:
    - filtered_points: отфильтрованные точки
    - filtered_labels: новые метки кластеров (с перенумерацией)
    - cluster_stats: статистика по кластерам (исходный_label: количество_точек)
    """
    # Считаем количество точек в каждом кластере
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Создаем маску для валидных кластеров
    valid_mask = (counts >= min_points)
    if max_points is not None:
        valid_mask &= (counts <= max_points)

    # Получаем метки кластеров, прошедших фильтрацию
    valid_labels = unique_labels[valid_mask]

    # Создаем маску для точек принадлежащих валидным кластерам
    points_mask = np.isin(labels, valid_labels)

    # Фильтруем точки
    filtered_points = points[points_mask]
    filtered_labels = labels[points_mask]

    # Перенумеровываем метки от 0 до N
    for new_label, old_label in enumerate(valid_labels):
        filtered_labels[filtered_labels == old_label] = new_label

    # Собираем статистику
    cluster_stats = dict(zip(unique_labels, counts))

    return filtered_points, filtered_labels, cluster_stats


def calculate_circularity( points_2d):
    """Метрика круговости через SVD"""
    _, S, _ = np.linalg.svd(points_2d - np.mean(points_2d, axis=0))
    return min(S) / max(S)  # 1.0 для идеальной окружности

def find_cluster_center_2d( points_2d):
    """Находит центр кластера в 2D через медиану (устойчиво к выбросам)"""
    return np.median(points_2d, axis=0)

def filter_by_circularity(points_2d, labels):
    """Фильтрация кластеров по форме"""
    good_labels = []
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster = points_2d[labels == label]
        circularity = calculate_circularity(cluster)
        if circularity >= 0.0:
            good_labels.append(label)
    return good_labels


def points_to_image(points, image_size=640, padding=0.1, point_size=1):
    """
    Преобразует массив точек в изображение с возможностью настройки размера точек.
    Возвращает 3-канальное RGB изображение для YOLO.
    """
    xy_points = points[:, :2]
    min_coords = np.min(xy_points, axis=0)
    max_coords = np.max(xy_points, axis=0)

    # Добавляем padding
    range_x, range_y = max_coords - min_coords
    min_coords -= [range_x * padding, range_y * padding]
    max_coords += [range_x * padding, range_y * padding]

    # Масштабирование
    scale = image_size / max(max_coords - min_coords)
    offset = min_coords

    # Создаем белое изображение (3 канала - RGB)
    image = np.full((image_size, image_size, 3), 255, dtype=np.uint8)

    # Преобразуем координаты
    scaled_points = (xy_points - offset) * scale
    pixel_coords = np.round(scaled_points).astype(int)

    # Рисуем точки с заданным размером (черным цветом)
    for x, y in pixel_coords:
        if 0 <= x < image_size and 0 <= y < image_size:
            if point_size == 1:
                image[y, x] = [0, 0, 0]  # Черный пиксель в RGB
            else:
                cv2.circle(image, (x, y), point_size, (0, 0, 0), -1)

    transform = {'offset': offset, 'scale': scale, 'image_size': image_size}
    return image, transform


def find_circle_with_yolo(image, model):
    """
    Обрабатывает картинку и находит окружность с помощью обученной модели YOLO.
    Возвращает (x, y, radius) в пиксельных координатах.
    """
    # Убедимся, что изображение в правильном формате для YOLO
    if len(image.shape) == 2:  # Если grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:  # Если одноканальное
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Предсказание с помощью YOLO
    results = model(image, conf=0.3, verbose=False)  # conf - порог уверенности

    if len(results[0].boxes) == 0:
        return None  # Не найдено окружностей

    # Берем первое обнаружение (самое уверенное)
    box = results[0].boxes[0]

    # Получаем координаты bounding box
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

    # Вычисляем центр и радиус окружности
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    radius = max((x2 - x1) / 2, (y2 - y1) / 2)

    return (center_x, center_y, radius)


def process_point_cloud(points, model, image_size=640, padding=0, point_size=2):
    """
    Полный pipeline обработки облака точек: преобразование в изображение,
    обнаружение окружности и преобразование координат.
    """
    # Преобразуем точки в изображение
    image, transformation = points_to_image(
        points, image_size=image_size, padding=padding, point_size=point_size
    )

    # Находим окружность и преобразуем координаты
    circle_coords = process_image(image, transformation, model)

    return circle_coords, image, transformation


def process_image(image, transformation: dict, model):
    """
    Находит окружность на изображении и преобразует координаты
    из пиксельной системы в декартовую.
    """
    # Находим окружность на изображении
    circle_result = find_circle_with_yolo(image, model)

    if circle_result is None:
        return None  # Окружность не найдена

    center_x_px, center_y_px, radius_px = circle_result

    # Преобразуем координаты центра из пикселей в декартовы
    center_x_coord, center_y_coord = pixel_to_coords(center_x_px, center_y_px, transformation)

    # Преобразуем радиус из пикселей в реальные единицы
    radius_coord = radius_px / transformation['scale']

    return (center_x_coord, center_y_coord, radius_coord)


def convert_layer_to_dataset1(filename, output_dir="cluster_images"):
    """
    Конвертирует NPZ файл с кластерами в изображения 640x640

    Параметры:
    - filename: путь к NPZ файлу
    - output_dir: директория для сохранения изображений
    """
    # Создаем директорию для сохранения
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем данные
    data = np.load(filename)
    points = data['points']
    labels = data['labels']

    # Фильтруем по circularity
    good_labels = filter_by_circularity(points, labels)

    print(f"Найдено {len(good_labels)} кластеров для обработки")

    for label in tqdm(good_labels, desc="Создание изображений"):
        # Получаем точки кластера
        cluster_points = points[labels == label]

        # Создаем изображение
        image, transform = points_to_image(cluster_points, image_size=640, point_size=2)

        # Сохраняем изображение
        output_filename = os.path.join(output_dir, f"cluster_{label:04d}.png")
        cv2.imwrite(output_filename, image)
        '''
        # Можно также сохранить метаданные для обратного преобразования
        np.savez_compressed(
            os.path.join(output_dir, f"cluster_{label:04d}_meta.npz"),
            transform=transform,
            point_count=len(cluster_points),
            original_label=label
        )'''

    print(f"Изображения сохранены в директорию: {output_dir}")


def create_shapefile_geopandas(results, filename="results/yolo_1try.shp"):
    """Создание Shapefile с помощью GeoPandas"""
    geometries = []
    attributes = []

    for i, result in enumerate(results):
        # Создаем круг как полигон
        center = Point(result['center_x'], result['center_y'])
        circle = center.buffer(result['radius'])

        geometries.append(circle)
        attributes.append({
            'id': i,
            'label': result['label'],
            'x': result['center_x'],
            'y': result['center_y'],
            'd': result['radius']*2,
        })

    # Создаем GeoDataFrame
    gdf = gpd.GeoDataFrame(attributes, geometry=geometries)

    # Устанавливаем систему координат (WGS84)
    gdf.crs = CRS.from_epsg(32639)  # WGS84

    # Сохраняем в Shapefile
    gdf.to_file(filename, encoding='utf-8')

    return 0

    # Создаем Shapefile

if __name__ == "__main__":
    # Загрузка модели YOLO
    YOLO_MODEL_PATH = "runs/detect/shapes_detection/weights/best.pt"
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("модель загружена")

    # Загрузка данных
    filename = "layers/clusters1.35-1.55.npz"
    data = np.load(filename)
    points = data['points']
    labels = data['labels']
    print('данные загружены, фильтруем по кругловатости')

    # Фильтруем по circularity
    good_labels = labels
        #filter_by_circularity(points, labels))
    print(f"Найдено {len(good_labels)} кластеров для обработки")

    # Список для результатов
    results = []

    for label in tqdm(list(set(good_labels)), desc="Создание изображений"):
        # Получаем точки кластера
        cluster_points = points[labels == label]

        # Обрабатываем кластер
        circle_coords, image, transformation = process_point_cloud(cluster_points, yolo_model)

        if circle_coords:
            results.append({
                'label': label,
                'center_x': circle_coords[0],
                'center_y': circle_coords[1],
                'radius': circle_coords[2],
                'points_count': len(cluster_points)
            })

            # Сохраняем изображение для отладки (опционально)
            #cv2.imwrite(f"debug/cluster_{label}_detection.png", image)

    # Вывод результатов
    print(f"\nОбработано кластеров: {len(results)}")
    for result in results:
        print(f"Кластер {result['label']}: центр ({result['center_x']:.2f}, {result['center_y']:.2f}), "
              f"радиус {result['radius']:.2f}, точек: {result['points_count']}")
    create_shapefile_geopandas(results,"results/yolo1.5-1.7.shp")