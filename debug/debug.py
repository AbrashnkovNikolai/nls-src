import numpy as np
import laspy
import os

def get_p_format_and_header(self, las_path):
    las = laspy.read(las_path)
    return las.point_format, las.header


def save_clusters(self, points, labels, filename="clusters.npz"):
    """Сохранение кластеризованных точек"""
    np.savez_compressed(
        os.path.join(self.output_dir, filename),
        points=points,  # Все точки (координаты X,Y,Z)
        labels=labels  # Метки кластеров (-1 для шума)
    )


def load_clusters(self, filename="clusters.npz"):
    """Загрузка ранее сохраненных кластеров"""
    data = np.load(os.path.join(self.output_dir, filename))
    return data['points'], data['labels']


def write_clusters_to_las(self, points, out_file_path: str, cluster_labels):
    if len(points) > 3:
        points = points.transpose()
    print("запись кластеров в las пошла")
    point_format, header = self.get_p_format_and_header(self.las_path)
    file_version = str(header.version)
    new_las = laspy.create(
        point_format=point_format,
        file_version=file_version
    )
    new_las.x = points[0]
    new_las.y = points[1]
    #new_las.z = points[2]

    r, g, b = zip(*[self.id_to_rgb(id) for id in cluster_labels])
    new_las.red = np.array(r)
    new_las.green = np.array(g)
    new_las.blue = np.array(b)
    new_las.write(out_file_path)
    print('запись кластеров в лас выполнена ')
