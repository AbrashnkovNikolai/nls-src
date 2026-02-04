# Для работы с массивами
import os

import numpy as np

# Для работы с las-файлами
import laspy

from las_functions import las_to_points

import open3d as o3d


def align_bounding_box_with_axes(points):
    """
    Выравнивает ограничивающий прямоугольник массива 3D точек
    параллельно осям координат с использованием PCA.

    Аргументы:
        points (np.array): Массив точек Nx3, где N - количество точек.

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

    return points_aligned # , rotation_matrix.T, centroid, min_coords_rotated


def main():
    pp = 1

    las_file_path = f'pp20-120-center-0.5-50_FSCT_output/DTM.las'
    las = laspy.read(las_file_path)
    points = las_to_points(las).transpose()

    interval = 50
    directory = f'C:\\Users\\lalay\\OneDrive\\Документы\\GitHub\\NLSgitdd\\pp20-120-center-0.5-50_FSCT_output'
    file_paths = []
    for item in os.listdir(directory):
        if item.lower().endswith('.las'):
            full_path = os.path.join(directory, item)
            if os.path.isfile(full_path):
                file_paths.append(full_path)
    for las_file_path in file_paths:

        # las_file_path = f'C:\\Users\\arsen\\Desktop\\python\\PyCUDATest\\pp{pp}-vls-spring-norm-no-noise-ground.las'
        las = laspy.read(las_file_path)
        points = np.vstack((points, las_to_points(las).transpose()))

    points = points.transpose()

    print(f'количество точек = {len(points[0])}')
    print(f'min x = {points[0].min()}')
    print(f'min y = {points[1].min()}')
    print(f'min z = {points[2].min()}')

    points[0] -= points[0].min()
    points[1] -= points[1].min()
    points[2] -= points[2].min()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.transpose())

    zeros = [points[0].min() - 1, points[1].min() - 1, 0]
    print(zeros)

    points = np.asarray(pcd.points).transpose()

    zeros = [points[0].min() - 1, points[1].min() - 1, 0]
    print(zeros)

    xmin = zeros[0]
    ymin = zeros[1]

    lines_points = [
        [points[0].min() - 1, points[1].min() - 1, 0],
        [points[0].max(), points[1].min() - 1, 0],
        [points[0].min() - 1, points[1].max(), 0],
        [points[0].min() - 1, points[1].min() - 1, interval * int(np.ceil(points[2].max() / interval))]
    ]

    lines = [(0, 1), (0, 2), (0, 3)]

    mark_lines_points = []
    mark_lines = []

    for i in range(1, int(np.ceil(points[0].max() / interval)) + 1):
        mark_lines_points.append([xmin + interval * i, points[1].max(), 0])
        mark_lines_points.append([xmin + interval * i, ymin - 1, 0])

        mark_lines.append([len(mark_lines_points) - 2, len(mark_lines_points) - 1])

    for i in range(1, int(np.ceil(points[1].max() / interval)) + 1):
        mark_lines_points.append([points[0].max(), ymin + interval * i, 0])
        mark_lines_points.append([xmin - 1, ymin + interval * i, 0])

        mark_lines.append([len(mark_lines_points) - 2, len(mark_lines_points) - 1])

    for i in range(1, int(np.ceil(points[2].max() / interval)) + 1):
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

    center = pcd.get_center()

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=las_file_path,
        width=800,
        height=600
    )

    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.add_geometry(mark_lines_set)

    vis.poll_events()
    vis.update_renderer()

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()



















