from typing import Tuple

from imports import *
import gc
def calculate_segment_heights_from_dict( segments_dict: dict, res_items: list) -> list:
    """
    Вычисляет высоту каждого сегмента и записывает в соответствующий словарь.
    Предполагает, что segment_id соответствует индексу в res_items.

    Args:
        segments_dict: Словарь {segment_id: массив точек}
        res_items: Список словарей с информацией о центрах

    Returns:
        Обновленный список словарей с высотами
    """
    dict_res_items = []
    for item in res_items:
        x, y, r = item
        res_item = {
            'x': x,
            'y': y,
            'z': 1.3,
            'd': r,
            's': 0,
            'h': 0,
        }
        dict_res_items.append(res_item)
    if not segments_dict or not res_items:
        return res_items

    for segment_id, points in segments_dict.items():
        if len(points) == 0:
            continue

        # Проверяем, что segment_id существует в res_items
        if segment_id >= len(res_items):
            print(f"Предупреждение: segment_id {segment_id} выходит за пределы res_items")
            continue

        # Вычисляем высоту сегмента (разница между max и min Z)
        z_coords = points[:, 2]
        min_z = np.min(z_coords)
        max_z = np.max(z_coords)
        height = max_z - min_z

        # Записываем высоту в соответствующий словарь
        dict_res_items[segment_id]['h'] = float(height)

    return dict_res_items


def segment_points_by_cylinders_old( points: np.ndarray,
                                centers: List[tuple],
                                min_height: float = config['min_height'],
                                min_points_in_segment: int = 1000,
                                buffer = 3) -> Dict[int, np.ndarray]:
    """
    Сегментирует облако точек по цилиндрическим областям вокруг центров.
    Использует KDTree для быстрого поиска.

    """
    if points is None or len(points) == 0:
        print("Нет точек для сегментации")
        return {}

    # Создаем KDTree для быстрого поиска (только XY координаты)
    tree = KDTree(points[:, :2])

    segments = []


    for i, center_data in enumerate(centers):
        # Извлекаем x, y, radius из кортежа и преобразуем в float
        x = float(center_data[0])
        y = float(center_data[1])
        radius = float(center_data[2])+buffer
        z_center = 1.3  # Значение по умолчанию для Z

        # Находим индексы точек в пределах радиуса в XY-плоскости
        center_xy = np.array([x, y])
        indices = tree.query_ball_point(center_xy, radius)
        if i % 200 == 0:
            gc.collect()


        candidate_points = points[indices]

        # Если задана высота, фильтруем по Z координате
        if min_height is not None:
            z_values = candidate_points[:, 2].astype(float)
            z_mask = np.abs(z_values - z_center) >= min_height / 2
            segments[i] = candidate_points[z_mask]
        else:
            segments[i] = candidate_points

    return segments


def segment_points_by_cylinders(points: np.ndarray,
                                centers: List[tuple],
                                min_height: float = config['min_height'],
                                min_points_in_segment: int = 1000,
                                buffer: float = 3) -> Tuple[Dict[int, np.ndarray], Dict[int, tuple]]:
    """
    Сегментирует облако точек по цилиндрическим областям вокруг центров.
    Возвращает словарь сегментов и соответствующих центров.
    """
    if points is None or len(points) == 0:
        print("Нет точек для сегментации")
        return {}, {}

    tree = KDTree(points[:, :2])
    segments = {}
    centers_dict = {}  # Новый словарь для соответствия сегмент-центр
    z_center = 1.3

    for i, center_data in enumerate(centers):
        x = float(center_data[0])
        y = float(center_data[1])
        radius = float(center_data[2]) + buffer

        center_xy = np.array([x, y])
        indices = tree.query_ball_point(center_xy, radius)

        if len(indices) < min_points_in_segment:
            continue

        indices_array = np.array(indices)
        candidate_points = points[indices_array]

        if min_height is not None:
            z_values = candidate_points[:, 2].astype(float)
            z_mask = np.abs(z_values - z_center) >= min_height / 2
            segment_points = candidate_points[z_mask]
        else:
            segment_points = candidate_points

        if len(segment_points) >= min_points_in_segment:
            segments[i] = segment_points
            centers_dict[i] = center_data  # Сохраняем центр для этого сегмента


        gc.collect()

    print(f"Создано {len(segments)} непустых сегментов")
    return segments, centers_dict


def save_segments_to_las(self,
                         segments: Dict[int, np.ndarray],
                         output_path: str):
    """
    Сохраняет сегментированные точки в LAS файл с разными цветами для каждого сегмента.

    Args:
        points: Все исходные точки (N, 3)
        segments: Словарь сегментов {id: точки сегмента}
        output_path: Путь для сохранения LAS файла
    """
    print("Сохранение сегментов в LAS файл...")

    # Получаем формат точек и заголовок из исходного файла
    point_format, header = self.get_p_format_and_header(self.las_path)
    file_version = str(header.version)

    # Создаем новый LAS файл
    new_las = laspy.create(
        point_format=point_format,
        file_version=file_version
    )

    all_points = []
    all_colors = []
    segment_ids = []

    # Генерируем случайные цвета для каждого сегмента
    unique_segment_ids = list(segments.keys())
    colors = {segment_id: self.id_to_rgb(segment_id) for segment_id in unique_segment_ids}

    # Собираем все точки с их цветами
    for segment_id, segment_points in segments.items():
        if len(segment_points) > 0:
            # Проверяем высоту сегмента
            z_coords = segment_points[:, 2]  # Предполагаем, что Z координата в третьей колонке
            min_z = np.min(z_coords)
            max_z = np.max(z_coords)
            height = max_z - min_z

            # Фильтруем по высоте
            if height >= 6.0:
                all_points.append(segment_points)
                # Создаем массив цветов для этого сегмента
                segment_colors = np.full((len(segment_points), 3), colors[segment_id])
                all_colors.append(segment_colors)
                segment_ids.extend([segment_id] * len(segment_points))
            else:
                # Можно добавить логирование для отсеянных сегментов
                print(f"Сегмент {segment_id} отсеян: высота {height:.2f} м < 6 м")

    if not all_points:
        print("Нет точек для сохранения")
        return

    # Объединяем все точки и цвета
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    # Записываем координаты
    new_las.x = all_points[:, 0]
    new_las.y = all_points[:, 1]
    new_las.z = all_points[:, 2]

    # Записываем цвета
    new_las.red = all_colors[:, 0]
    new_las.green = all_colors[:, 1]
    new_las.blue = all_colors[:, 2]

    # Сохраняем файл
    new_las.write(output_path)
    print(f"Сохранено {len(all_points)} точек в {output_path}")

    # Выводим информацию о каждом сегменте
    valid_segments = {k: v for k, v in segments.items() if len(v) > 0}
    print(f"Количество сегментов: {len(valid_segments)}")

    for segment_id, segment_points in valid_segments.items():
        z_coords = segment_points[:, 2]
        min_z = np.min(z_coords)
        max_z = np.max(z_coords)
        height = max_z - min_z
        print(f"Сегмент {segment_id}: {len(segment_points)} точек, высота: {height:.2f} м")
