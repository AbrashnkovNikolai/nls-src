
from imports import hdbscan,config,pd,tqdm,np
from megiddo import RobustCircleDetector
from functions.filters_and_metrics import delta_count_filter, filter_by_circularity


def clusterize_points(points,
        min_cluster_size=int(config['MIN_CLUSTER_SIZE']),
        cluster_selection_method=config['cluster_selection_method'],
        cluster_selection_epsilon=float(config['cluster_selection_epsilon']),
        core_dist_n_jobs=int(config['core_dist_n_jobs'])
        ):

    print("Кластеризация HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=core_dist_n_jobs
    )
    labels = clusterer.fit_predict(points)
    print(f"уникальных кластеров: {len(set(labels))}")
    return labels

def process_cluster_for_dbh( cluster_points, detector: RobustCircleDetector,
                    ):
        detector.clear_points()
        detector.add_points(cluster_points)
        if detector.compute_robust(coverage=0.98):
            center, radius = detector.get_circle()
            d = (radius * 2)
            if not (config['MIN_RADIUS'] <= radius <= config['MAX_RADIUS']):
                return None
            if delta_count_filter(cluster_points, center, radius):
                detector.clear_points()
                return {
                    'x': center[0],
                    'y': center[1],
                    'z': 1.3,
                    'd': d,
                    's': 0,
                    'h': 0,
                    #'geometry': Point(center[0], center[1]).buffer(radius)
                }
        detector.clear_points()

def process_layer( points_2d, isCached=False,cicle_threshold=0.4):
        """Обработка всего слоя точек"""
        if not isCached:
            labels = clusterize_points(points_2d)
            #save_clusters(points_2d,labels)
        else:
            #points_2d, labels = load_clusters()
            print(points_2d[:3])
            #write_clusters_to_las(points_2d,"colored_clusters.las",labels)
            print('кластеры не загружены')
        print(f'фильтр по форме круга : {cicle_threshold} ')
        good_labels = filter_by_circularity(points_2d, labels,cicle_threshold)
        cirk_detector = RobustCircleDetector()
        circles = []
        centers = []
        for label in tqdm(good_labels, desc="Обработка кластеров"):
            cluster_points = points_2d[labels == label]
            tree = process_cluster_for_dbh(cluster_points, cirk_detector)

            if tree:
                circles.append(tree)

        return pd.DataFrame(circles) if circles else None

################### процессинг кластероа в сегментах
def find_tree_height_by_reclustering(segment_points, max_interval=2.0):
    """
    Определение высоты дерева через повторную кластеризацию точек сегмента
"""

    # Используем твою функцию кластеризации для точек сегмента
    labels = clusterize_points(segment_points,cluster_selection_epsilon=0.2,min_cluster_size=500,core_dist_n_jobs=4)

    # Анализируем полученные кластеры
    unique_labels = np.unique(labels)
    valid_clusters = [label for label in unique_labels if label != -1]

    print(f"В сегменте найдено {len(valid_clusters)} кластеров")

    if len(valid_clusters) <= 1:  # только один кластер или шум
        return np.max(segment_points[:, 2])

    # Собираем информацию по каждому кластеру
    clusters_info = []
    for label in valid_clusters:
        cluster_mask = labels == label
        cluster_points = segment_points[cluster_mask]

        cluster_info = {
            'label': label,
            'point_count': len(cluster_points),
            'min_height': np.min(cluster_points[:, 2]),
            'max_height': np.max(cluster_points[:, 2]),
            'mean_height': np.mean(cluster_points[:, 2]),
            'center': np.mean(cluster_points[:, :2], axis=0)
        }
        clusters_info.append(cluster_info)

    # Сортируем кластеры по минимальной высоте (снизу вверх)
    clusters_info.sort(key=lambda x: x['min_height'])

    # Логика выбора правильного дерева:
    for i, cluster in enumerate(clusters_info):
        # Проверяем, есть ли над этим кластером другие с большим разрывом
        higher_clusters = []
        for j in range(i + 1, len(clusters_info)):
            higher_cluster = clusters_info[j]
            # Если разрыв между кластерами больше max_interval - считаем это разными деревьями
            if higher_cluster['min_height'] - cluster['max_height'] > max_interval:
                higher_clusters.append(higher_cluster)

        if higher_clusters:
            # Нашли кластеры значительно выше - возвращаем высоту текущего (нижнего) кластера
            print(f"Выбран кластер {cluster['label']} (высота: {cluster['max_height']:.1f}м), "
                  f"игнорируем {len(higher_clusters)} верхних кластеров")
            return cluster['max_height']

    # Если не нашли явных разделений, берем самый нижний кластер
    if clusters_info:
        return clusters_info[0]['max_height']

    # Fallback
    return np.max(segment_points[:, 2])


def process_segment_reclustering(seg_id, segment_points, centers_dict):
    """Обработка сегмента с повторной кластеризацией"""
    center = centers_dict[seg_id]
    x, y, radius = center

    tree_height = find_tree_height_by_reclustering(segment_points, segment_points)

    return {
        'x': center[0],
        'y': center[1],
        'z': tree_height,
        'd': center[2],
        's': 0,
        'h': tree_height,
    }
