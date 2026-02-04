import argparse
from shapely import Polygon
import json


def parse_parameters():
    parser = argparse.ArgumentParser(description='Обработка LiDAR данных для обнаружения деревьев')

    # Группа для путей
    paths_group = parser.add_argument_group('Пути к файлам')
    paths_group.add_argument('--las_path', type=str, default='pp20cutbuf5NORM-norm-nograss.las',
                             help='Путь до входного LAS файла (по умолчанию: pp20cutbuf5NORM-norm-nograss.las)')
    paths_group.add_argument('--output_dir', type=str, default='testovaya',
                             help='Выходная директория (по умолчанию: testovaya)')
    paths_group.add_argument('--output', type=str, default='last_pp20_circle.csv',
                             help='Название выходного CSV (по умолчанию: last_pp20_circle.csv)')
    paths_group.add_argument('--yolo_model_path', type=str, default='yolo_better_than_main.pt',
                             help='Путь до модели YOLO (по умолчанию: yolo_better_than_main.pt)')

    # Группа для параметров среза и радиусов
    processing_group = parser.add_argument_group('Параметры обработки')
    processing_group.add_argument('--layer_min', type=float, default=1.3,
                                  help='Минимальная граница среза по Z координате (по умолчанию: 1.3)')
    processing_group.add_argument('--layer_max', type=float, default=1.5,
                                  help='Максимальная граница среза по Z координате (по умолчанию: 1.5)')
    processing_group.add_argument('--max_radius', type=float, default=0.45,
                                  help='Максимальный радиус круга (по умолчанию: 0.45)')
    processing_group.add_argument('--min_radius', type=float, default=0.05,
                                  help='Минимальный радиус круга (по умолчанию: 0.05)')

    # Группа для кластеризации
    clustering_group = parser.add_argument_group('Параметры кластеризации')
    clustering_group.add_argument('--min_cluster_size', type=int, default=50,
                                  help='Минимальный размер кластера в срезе (по умолчанию: 50)')
    clustering_group.add_argument('--cluster_selection_method', type=str, default='eom',
                                  help='Метод выбора кластеров (по умолчанию: eom)')
    clustering_group.add_argument('--cluster_selection_epsilon', type=float, default=0.1,
                                  help='Эпсилон для выбора кластеров (по умолчанию: 0.1)')
    clustering_group.add_argument('--core_dist_n_jobs', type=int, default=8,
                                  help='Количество ядер для распараллеливания (по умолчанию: 8, -1: все ядра)')

    # Группа для фильтрации
    filtering_group = parser.add_argument_group('Параметры фильтрации')
    filtering_group.add_argument('--circularity_threshold', type=float, default=0.4,
                                 help='Порог кругловатости (1.0 - идеальный круг) (по умолчанию: 0.4)')
    filtering_group.add_argument('--coverage', type=float, default=0.98,
                                 help='Процент точек кластера покрывать кругом (по умолчанию: 0.98)')
    filtering_group.add_argument('--circle_thickness', type=float, default=0.03,
                                 help='Толщина окружности (по умолчанию: 0.03)')
    filtering_group.add_argument('--scale', type=float, default=0.25,
                                 help='Радиус внутреннего круга (в %% от найденного) (по умолчанию: 0.25)')
    filtering_group.add_argument('--delta', type=float, default=0.4,
                                 help='Допустимая разница в количестве между внутренним и найденным кругом (по умолчанию: 0.4)')

    # Группа для выделения стволов
    trunk_group = parser.add_argument_group('Параметры выделения стволов')
    trunk_group.add_argument('--min_height', type=float, default=6.0,
                             help='Минимальная высота ствола (по умолчанию: 6.0)')
    trunk_group.add_argument('--min_points_in_segment', type=int, default=1000,
                             help='Минимальное количество точек в сегменте (по умолчанию: 1000)')

    # Дополнительные опции
    parser.add_argument('--config', type=str,
                        help='Путь к JSON файлу с параметрами (переопределяет значения по умолчанию)')
    parser.add_argument('--border_coords', type=str,
                        help='Координаты полигона в формате "x1,y1;x2,y2;x3,y3;x4,y4"')

    args = parser.parse_args()

    # Обработка полигона границ
    if args.border_coords:
        # Парсинг координат из строки
        coords = []
        points = args.border_coords.split(';')
        for point in points:
            x, y = map(float, point.split(','))
            coords.append((x, y))
        border_polygon = Polygon(coords)
    else:
        # Используем полигон по умолчанию
        border_polygon = Polygon([
            (357430.910, 6758474.922),
            (357462.474, 6758520.326),
            (357550.327, 6758456.846),
            (357512.8682, 6758413.7518)
        ])

    # Загрузка параметров из конфигурационного файла если указан
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Обновляем аргументы значениями из конфига
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Собираем параметры в словарь (совместимый с твоим текущим форматом)
    parameters = {
        # пути
        'las_path': args.las_path,
        'output_dir': args.output_dir,
        'output': args.output,
        'yolo_model_path': args.yolo_model_path,
        'LAYER': (args.layer_min, args.layer_max),

        # радиусы
        'MAX_RADIUS': args.max_radius,
        'MIN_RADIUS': args.min_radius,

        # параметры кластеризации
        'MIN_CLUSTER_SIZE': args.min_cluster_size,
        'cluster_selection_method': args.cluster_selection_method,
        'cluster_selection_epsilon': args.cluster_selection_epsilon,
        'core_dist_n_jobs': args.core_dist_n_jobs,

        # параметры фильтрации
        'border_polygon': border_polygon,
        'CIRCULARITY_THRESHOLD': args.circularity_threshold,
        "coverage": args.coverage,
        'circle_thickness': args.circle_thickness,

        # фильтр по пустоте внутри круга
        'scale': args.scale,
        'delta': args.delta,

        # выделение стволов
        'min_height': args.min_height,
        'min_points_in_segment': args.min_points_in_segment,
    }

    return parameters

config = {
  "yolo_model_path": "yolo_better_than_main.pt",
  "las_path": "las_example.las",
  "etalon_path": "etalon_example.shp",
  "output":"last_output.csv",
  "LAYER": [1.3, 1.5],
  "MAX_RADIUS": 0.45,
  "MIN_RADIUS": 0.05,
  "MIN_CLUSTER_SIZE": 50,
  "cluster_selection_method": "eom",
  "cluster_selection_epsilon": 0.1,
  "core_dist_n_jobs": 8,
  "CIRCULARITY_THRESHOLD": 0.4,
  "coverage": 0.98,
  "circle_thickness": 0.03,
  "scale": 0.25,
  "delta": 0.4,
  "min_height": 6,
  "min_points_in_segment": 1000
}
