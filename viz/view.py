# Для работы с массивами
import numpy as np

# Для работы с las-файлами
import laspy
import open3d as o3d


def las_to_points(las: laspy.LasData):
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)

    # Получаем цвета из LAS-файла
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        red = np.array(las.red, dtype=np.float64) // 255.0
        green = np.array(las.green, dtype=np.float64) // 255.0
        blue = np.array(las.blue, dtype=np.float64) // 255.0

        colors = np.vstack((red, green, blue)).T
    else:
        # Если цветов нет, используем белый цвет для всех точек
        colors = np.ones((len(x), 3), dtype=np.float64)

    points = np.vstack((x, y, z)).T

    return points, colors


def align_bounding_box_with_axes(points):
    """
    Выравнивает ограничивающий прямоугольник массива 3D точек
    параллельно осям координат с использованием PCA.

    Аргументы:
        points2d (np.array): Массив точек Nx3, где N - количество точек.

    Возвращает:
        np.array: Массив точек Nx3, где ограничивающий прямоугольник выровнен по осям.
        np.array: Матрица поворота, примененная для выравнивания.
        np.array: Вектор трансляции (центроид).
        np.array: Вектор трансляции (для смещения в положительный октант).
    """
    if points.shape[1] != 3:
        raise ValueError("Точки должны быть 3D (форма Nx3)")

    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    covariance_matrix = np.cov(points_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    rotation_matrix = eigenvectors.T

    points_rotated = np.dot(points_centered, rotation_matrix.T)

    min_coords_rotated = np.min(points_rotated, axis=0)
    points_aligned = points_rotated - min_coords_rotated

    return points_aligned  # , rotation_matrix.T, centroid, min_coords_rotated


def main():
    #las_file_path = f'C:\\Users\\lalay\\OneDrive\\Документы\\GitHub\\NLSgitdd\\tree_detection_results\\vls-cut_FSCT_output/stem_points.las'
    las_file_path = 'viz_trees_cones.las'
    las = laspy.read(las_file_path)
    points, colors = las_to_points(las)
    print(colors[:3])
    interval = 5

    print(f'Путь = {las_file_path}')
    print(f'количество точек = {len(points)}')
    print(f'min x = {points[:, 0].min()}')
    print(f'min y = {points[:, 1].min()}')
    print(f'min z = {points[:, 2].min()}')
    print(f'Цвета доступны: {colors is not None}')

    points[:, 0] -= points[:, 0].min()
    points[:, 1] -= points[:, 1].min()
    points[:, 2] -= points[:, 2].min()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Устанавливаем цвета точек
    pcd.colors = o3d.utility.Vector3dVector(colors)

    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    covariance_matrix = np.cov(points_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    rotation_matrix = eigenvectors.T

    zeros = [points[:, 0].min() - 1, points[:, 1].min() - 1, 0]
    print(zeros)

    points_array = np.asarray(pcd.points)

    zeros = [points_array[:, 0].min() - 1, points_array[:, 1].min() - 1, 0]
    print(zeros)

    xmin = zeros[0]
    ymin = zeros[1]

    lines_points = [
        [points_array[:, 0].min() - 1, points_array[:, 1].min() - 1, 0],
        [points_array[:, 0].max(), points_array[:, 1].min() - 1, 0],
        [points_array[:, 0].min() - 1, points_array[:, 1].max(), 0],
        [points_array[:, 0].min() - 1, points_array[:, 1].min() - 1,
         interval * int(np.ceil(points_array[:, 2].max() / interval))]
    ]

    lines = [(0, 1), (0, 2), (0, 3)]

    mark_lines_points = []
    mark_lines = []

    for i in range(1, int(np.ceil(points_array[:, 0].max() / interval)) + 1):
        mark_lines_points.append([xmin + interval * i, points_array[:, 1].max(), 0])
        mark_lines_points.append([xmin + interval * i, ymin - 1, 0])

        mark_lines.append([len(mark_lines_points) - 2, len(mark_lines_points) - 1])

    for i in range(1, int(np.ceil(points_array[:, 1].max() / interval)) + 1):
        mark_lines_points.append([points_array[:, 0].max(), ymin + interval * i, 0])
        mark_lines_points.append([xmin - 1, ymin + interval * i, 0])

        mark_lines.append([len(mark_lines_points) - 2, len(mark_lines_points) - 1])

    for i in range(1, int(np.ceil(points_array[:, 2].max() / interval)) + 1):
        mark_lines_points.append([xmin, ymin, interval * i])
        mark_lines_points.append([xmin, ymin + 2, interval * i])
        mark_lines_points.append([xmin + 2, ymin, interval * i])

        mark_lines.append([len(mark_lines_points) - 3, len(mark_lines_points) - 2])
        mark_lines.append([len(mark_lines_points) - 3, len(mark_lines_points) - 1])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(lines_points),
        lines=o3d.utility.Vector2iVector(lines)
    )

    mark_lines_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(mark_lines_points),
        lines=o3d.utility.Vector2iVector(mark_lines)
    )

    # Устанавливаем цвет для линий (белый)
    line_set.colors = o3d.utility.Vector3dVector([[1, 1, 1] for _ in range(len(lines))])
    mark_lines_set.colors = o3d.utility.Vector3dVector([[1, 1, 1] for _ in range(len(mark_lines))])

    center = pcd.get_center()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=las_file_path, width=800, height=600)
    #render_option = vis.get_render_option()
    #render_option.background_color = np.array([0.0, 0.0, 0.0])  # RGB в диапазоне [0, 1]

    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.add_geometry(mark_lines_set)

    # Для создания скриншотов,
    # если отключить vis.run окно закроется и можно делать следующий кадр
    # vis.capture_screen_image(f'screen1.png', True)

    # vis.poll_events()
    # vis.update_renderer()

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()