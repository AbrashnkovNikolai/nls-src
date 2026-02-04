import gc
from typing import Tuple, Optional

from imports import *
from scipy.spatial import KDTree


def id_to_rgb(cluster_id):
    # Преобразуем cluster_id в строку и хешируем (SHA-256)
    hash_bytes = hashlib.sha256(str(cluster_id).encode()).digest()

    # Берем первые 3 байта хеша для R, G, B
    r = hash_bytes[0] % 256
    g = hash_bytes[1] % 256
    b = hash_bytes[2] % 256

    return r, g, b

def calculate_circularity(points_2d):
    """Метрика круговости через SVD"""
    _, S, _ = np.linalg.svd(points_2d - np.mean(points_2d, axis=0))
    return min(S) / max(S)  # 1.0 для идеальной окружности

def find_cluster_center_2d(points_2d):
    """Находит центр кластера в 2D через медиану (устойчиво к выбросам)"""
    return np.median(points_2d, axis=0)

def filter_by_circularity(points_2d, labels,CIRCULARITY_THRESHOLD):
    """Фильтрация кластеров по форме"""
    good_labels = []
    for label in np.unique(labels):
        if label == -1:
            continue

        cluster = points_2d[labels == label]
        circularity = calculate_circularity(cluster)
        if circularity >= CIRCULARITY_THRESHOLD:
            good_labels.append(label)
    return good_labels

def filter_by_border(circles_centers: list[Point], border_polygon: Polygon):
    insides = []
    for x, y, r in circles_centers:

        point = Point(x, y)
        if border_polygon.contains(point):
            insides.append((x, y, r))
    return insides


def delta_count_filter(points, center, radius, scale=config['scale'], delta=config['delta']):
    """
    Фильтр по количеству между двумя окружностями в одной точке с радиусом  r и r*scale, \
    returns: bool является ли круг пустым внутри
    """
    tree = KDTree(points)
    count_main = len(list(tree.query_ball_point(center, radius)))
    count_inner = len(list(tree.query_ball_point(center, radius * scale)))
    if count_inner == 0:
        return True
    if count_main < 30:
        return False
    return (abs(count_main - count_inner) / count_inner) < delta


def lmf_point_cloud(seg, radius):
    if len(seg) == 0:
        return 0

    points_2d = seg[:, :2]
    tree = KDTree(points_2d)
    neighbors_indices = tree.query_ball_point(points_2d, radius)

    lmf_values = []
    for i, indices_list in enumerate(neighbors_indices):
        if indices_list:
            indices_arr = np.array(indices_list, dtype=int)
            neighbor_points = seg[indices_arr]
            height_diff = np.abs(neighbor_points[:, 2] - seg[i][2])
            count_within_1 = np.sum(height_diff <= 1)
            lmf = count_within_1 / len(neighbor_points)
        else:
            lmf = 0
        lmf_values.append(lmf)
        gc.collect()

    # Возвращаем среднее значение LMF для всего облака
    return np.max(lmf_values) if lmf_values else 0


def find_highest_local_maximum_large(
        points_array: np.ndarray,
        neighborhood_radius: float,
        batch_size: int = 50000  # Уменьшил размер батча для кэша
) -> Tuple[Optional[Tuple[float, float, float]], float]:
    if len(points_array) == 0:
        return None, -np.inf

    n_points = len(points_array)

    # Создаем KD-tree
    xy_points = points_array[:, :2].astype(np.float32)
    tree = KDTree(xy_points)

    # Находим ВСЕХ соседей для всех точек заранее
    print("Поиск соседей...")
    all_neighbors = tree.query_ball_tree(tree, neighborhood_radius)

    local_maxima = []

    # Проверяем локальные максимумы
    print("Поиск локальных максимумов...")
    for i in range(n_points):
        neighbors_indices = all_neighbors[i]
        current_z = points_array[i, 2]

        # Пропускаем точку без соседей
        if not neighbors_indices:
            continue

        # Проверяем, что текущая точка - максимум среди соседей
        neighbors_z = points_array[neighbors_indices, 2]
        if np.max(neighbors_z) == current_z:

            max_indices = neighbors_indices[np.where(neighbors_z == current_z)[0]]
            if i == max_indices[0]:  # Добавляем только если мы первая из точек с максимальной высотой
                local_maxima.append(points_array[i])

    # Обработка результатов
    if not local_maxima:
        max_idx = np.argmax(points_array[:, 2])
        max_point = points_array[max_idx]
        return tuple(max_point), float(max_point[2])

    # Находим самый высокий локальный максимум
    local_maxima_array = np.array(local_maxima)
    max_idx = np.argmax(local_maxima_array[:, 2])
    highest_point = local_maxima_array[max_idx]

    return tuple(highest_point), float(highest_point[2])


def find_highest_near_axis_priority_height(points_array: np.ndarray,
                                           cylinder_axis_point: Tuple[float, float, float],
                                           max_distance: float = None,
                                           height_weight: float = 0.7) -> Tuple[
    Optional[Tuple[float, float, float]], float]:
    """
    Упрощенная версия для вертикального цилиндра.
    """
    if len(points_array) == 0:
        return None, -np.inf

    axis_x, axis_y, axis_z = cylinder_axis_point

    # Для вертикального цилиндра расстояние до оси вычисляется в XY плоскости
    distances = np.sqrt((points_array[:, 0] - axis_x) ** 2 + (points_array[:, 1] - axis_y) ** 2)
    heights = points_array[:, 2]

    # Фильтрация по расстоянию
    if max_distance is not None:
        mask = distances <= max_distance
        points_array = points_array[mask]
        distances = distances[mask]
        heights = heights[mask]

        if len(points_array) == 0:
            return None, -np.inf

    # Комбинированная оценка
    inv_distances = 1.0 / (distances + 1e-8)
    scores = height_weight * heights + (1 - height_weight) * inv_distances

    best_idx = np.argmax(scores)
    best_point = points_array[best_idx]

    return tuple(best_point), float(best_point[2])
def find_closest_pairs(elevation_points, trunk_points):
    """
    Находит для каждой высотной точки ближайшую точку ствола.

    Args:
        elevation_points: numpy array высотных точек shape (n, 3)
        trunk_points: numpy array точек стволов shape (m, 3)

    Returns:
        list: список кортежей [(elev_point, trunk_point, distance), ...]
    """
    tree = cKDTree(trunk_points)
    distances, indices = tree.query(elevation_points)

    pairs = []
    for i in range(len(elevation_points)):
        elev_point = elevation_points[i]
        trunk_point = trunk_points[indices[i]]
        distance = float(distances[i])
        pairs.append((elev_point, trunk_point))

    return pairs