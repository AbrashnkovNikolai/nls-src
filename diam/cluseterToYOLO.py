import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from shapely.geometry import Point
import geopandas as gpd


def detect_circles(image):
    """Находит круги на изображении с помощью HoughCircles."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )
    return circles[0] if circles is not None else []


def prepare_yolo_dataset(dataset_path, output_dir="yolo_dataset"):
    """Создает датасет для YOLO с автоматической разметкой кругов."""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    for img_name in os.listdir(dataset_path):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Детекция кругов
        #
        circles = detect_circles(img)
        if len(circles) == 0:
            continue

        # Сохранение изображения
        output_img_path = os.path.join(output_dir, "images", img_name)
        cv2.imwrite(output_img_path, img)

        # Создание разметки YOLO
        label_path = os.path.join(output_dir, "labels", os.path.splitext(img_name)[0] + ".txt")
        h, w = img.shape[:2]

        with open(label_path, 'w') as f:
            for circle in circles:
                x, y, r = circle
                # Конвертация в формат YOLO: класс x_center y_center width height (нормализованные)
                x_center = x / w
                y_center = y / h
                width = (2 * r) / w
                height = (2 * r) / h
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def train_yolo_model(data_dir="yolo_dataset"):
    """Обучает модель YOLO"""
    # Создаем конфиг для YOLO
    yaml_path = os.path.join(data_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(data_dir)}\n"
                f"train: images\n"
                f"val: images\n\n"
                f"names:\n  0: circle\n"
                f"nc: 1")

    # Инициализируем модель
    model = YOLO("yolov8n.pt")  # или yolov8s.pt для меньшей модели

    # Обучаем модель
    results = model.train(
        data=yaml_path,
        epochs=20,
        imgsz=640,
        batch=8,
        patience=10,
        device='cuda' if YOLO('yolov8n.pt').device.type == 'cuda' else 'cpu',
        name="circle_detector"
    )

    return model


def points_to_image(points, image_size=640, padding=0.1, point_size=4):
    """Преобразует массив точек в изображение для YOLO"""
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

    # Создаем белое изображение
    image = np.full((image_size, image_size), 255, dtype=np.uint8)

    # Преобразуем координаты
    scaled_points = (xy_points - offset) * scale
    pixel_coords = np.round(scaled_points).astype(int)

    # Рисуем точки
    for x, y in pixel_coords:
        if 0 <= x < image_size and 0 <= y < image_size:
            image[y, x] = [0,0,0]

    transform = {'offset': offset, 'scale': scale, 'image_size': image_size}
    return image, transform

def detect_with_yolo(model, points, labels, output_dir="detections"):
    """Обнаруживает стволы с помощью YOLO"""
    os.makedirs(output_dir, exist_ok=True)
    detections = []
    for label in np.unique(labels):
        if label == -1: continue

        cluster = points[labels == label]
        if len(cluster) < 50: continue

        img, transform = points_to_image(cluster)
        img_path = os.path.join(output_dir, f"cluster_{label}.png")
        Image.fromarray(img).save(img_path)

        # Детекция
        results = model(img_path)

        # Обработка результатов
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                width, height = x2 - x1, y2 - y1
                radius_px = (width + height) / 4  # Средний "радиус" в пикселях
                radius_m = radius_px / transform['scale']  # Переводим в метры
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

                # Преобразуем координаты обратно
                real_coords = (np.array([center_x, center_y]) / transform['scale']) + transform['offset']

                detections.append({
                    'cluster_label': int(label),
                    'center_px': (float(center_x), float(center_y)),
                    'center_coords': real_coords.tolist(),
                    'width_px': float(x2 - x1),
                    'height_px': float(y2 - y1),
                    'confidence': float(box.conf[0]),
                    'radius': radius_m
                })

    return detections

def save_to_shapefile(detections, output_path="detections.shp"):
    """Сохраняет результаты в Shapefile"""
    gdf = gpd.GeoDataFrame(
        detections,
        geometry=[Point(d['center_coords']).buffer(d['radius']) for d in detections],
        crs="EPSG:32639"  # Укажите вашу систему координат
    )
    gdf.to_file(output_path)

def main():
    # Путь к вашему датасету с изображениями
    dataset_path = r"C:\Users\lalay\.cache\kagglehub\datasets\aman2000jaiswal\circle-object-detection\versions\2"

    # 1. Подготовка датасета
    print("Подготовка датасета...")
    #prepare_yolo_dataset(dataset_path)
    data = np.load('tree_detection_results/clusters.npz')
    points, labels = data['points'], data['labels']
    # 2. Обучение модели
    print("Обучение модели...")
    #model = train_yolo_model()
    model = YOLO('runs/detect/circle_detector2/weights/best.pt')
    # 3. Сохранение обученной модели
    #model.save("best_circle_detector.pt")

    # 4. Детекция
    detections = detect_with_yolo(model, points, labels)

    # 5. Сохранение результатов
    save_to_shapefile(detections)
    print(f"Обнаружено {len(detections)} стволов, сохранено в detections.shp")

if __name__ == "__main__":
    main()