# Для работы с массивами
import os

import numpy as np

from skimage import exposure
import cv2

# Для работы с las-файлами
import laspy

from las_functions import las_to_points

import open3d as o3d

state = -1  # все точки видны


def gen_axes_lines(points, interval=5):
    axes_lines_points = [
        [points[0].min() - interval, points[1].min() - interval, 0],
        [points[0].max(), points[1].min() - interval, 0],
        [points[0].min() - interval, points[1].max(), 0],
        [points[0].min() - interval, points[1].min() - interval, interval * int(np.ceil(points[2].max() / interval))]
    ]
    axes_lines = [(0, 1), (0, 2), (0, 3)]
    return o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axes_lines_points),
        lines=o3d.utility.Vector2iVector(axes_lines)
    )


def gen_lines(points, interval=5):
    zeros = [points[0].min() - interval, points[1].min() - interval, 0]
    xmin = zeros[0]
    ymin = zeros[1]

    mark_lines_points = []
    mark_lines = []

    for i in range(0, int(np.ceil(points[0].max() / interval)) + 1):
        mark_lines_points.append([points[0].min() + interval * i, points[1].max(), 0])
        mark_lines_points.append([points[0].min() + interval * i, ymin - 1, 0])

        mark_lines.append([len(mark_lines_points) - 2, len(mark_lines_points) - 1])

    for i in range(0, int(np.ceil(points[1].max() / interval)) + 1):
        mark_lines_points.append([points[0].max(), points[1].min() + interval * i, 0])
        mark_lines_points.append([xmin - 1, points[1].min() + interval * i, 0])

        mark_lines.append([len(mark_lines_points) - 2, len(mark_lines_points) - 1])

    for i in range(1, int(np.ceil(points[2].max() / interval)) + 1):
        mark_lines_points.append([xmin, ymin, interval * i])
        mark_lines_points.append([xmin, ymin + 2, interval * i])
        mark_lines_points.append([xmin + 2, ymin, interval * i])

        mark_lines.append([len(mark_lines_points) - 3, len(mark_lines_points) - 2])
        mark_lines.append([len(mark_lines_points) - 3, len(mark_lines_points) - 1])

    return o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(mark_lines_points),
        lines=o3d.utility.Vector2iVector(mark_lines)
    )


def gen_mark_text(points, interval=5, size=0.1, color=None):
    if color is None:
        color = [1, 0, 0]
    xmin = points[0].min() - interval
    ymin = points[1].min() - interval

    labels = []

    for i in range(0, int(np.ceil(points[0].max() / interval)) + 1):
        pos = np.array([points[0].min() + interval * i, ymin, 0])

        text_mesh = o3d.t.geometry.TriangleMesh.create_text(text=f'{i * interval}', depth=0.3).to_legacy()
        text_mesh.translate(pos)
        text_mesh.paint_uniform_color(color)
        text_mesh.scale(size, center=pos)
        text_mesh.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 3 * np.pi / 2]),
            center=pos
        )
        labels.append(text_mesh)

    for i in range(0, int(np.ceil(points[1].max() / interval)) + 1):
        pos = np.array([xmin - len(f'{i * interval}'), points[1].min() + interval * i, 0])

        text_mesh = o3d.t.geometry.TriangleMesh.create_text(text=f'{i * interval}', depth=0.3).to_legacy()
        text_mesh.translate(pos)
        text_mesh.paint_uniform_color(color)
        text_mesh.scale(size, center=pos)
        text_mesh.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 0]),
            center=pos
        )
        labels.append(text_mesh)

    for i in range(0, int(np.ceil(points[2].max() / interval)) + 1):
        pos = np.array([xmin, ymin, interval * i])

        text_mesh = o3d.t.geometry.TriangleMesh.create_text(text=f'{i * interval}', depth=0.3).to_legacy()
        text_mesh.translate(pos)
        text_mesh.paint_uniform_color(color)
        text_mesh.scale(size, center=pos)
        text_mesh.rotate(
            o3d.geometry.get_rotation_matrix_from_xyz([np.pi / 2, 0, 0]),
            center=pos
        )
        labels.append(text_mesh)

    return labels


# noinspection PyTypeChecker
def view(
        name: str,
        points_list: list,
        interval=5,
        point_colors_list: list = None,
        const_points_list=None
):
    points = points_list[0].transpose()
    for p in points_list[1:]:
        points = np.vstack((points, p.transpose()))

    points = points.transpose()

    min_x = points[0].min()
    min_y = points[1].min()
    min_z = points[2].min()

    print(f'количество точек = {len(points[0])}')
    print(f'min x = {min_x}')
    print(f'min y = {min_y}')
    print(f'min z = {min_z}')

    points[0] -= points[0].min()
    points[1] -= points[1].min()
    points[2] -= points[2].min()

    const_pcd = o3d.geometry.PointCloud()

    if const_points_list:
        const_points = const_points_list[0]
        for p in const_points_list[1:]:
            const_points = np.vstack((const_points, p.transpose()))

        const_points[0] -= min_x
        const_points[1] -= min_y
        # const_points[2] -= min_z

        const_pcd.points = o3d.utility.Vector3dVector(const_points.transpose())
        const_pcd.colors = o3d.utility.Vector3dVector(
            np.full((len(const_points[0]), 3), [0, 0, 0])
        )

    state_to_points = {}

    for i in range(0, len(points_list)):
        p = points_list[i].copy()

        p[0] -= min_x
        p[1] -= min_y

        p = o3d.utility.Vector3dVector(p.transpose())
        state_to_points.setdefault(i, {'points': p, 'color': None})

        if point_colors_list is not None:
            colors = o3d.utility.Vector3dVector(
                np.full((len(points_list[i][0]), 3), point_colors_list[i])
            )
            state_to_points[i]['color'] = colors

    state_to_points.setdefault(-1, {
            'points': o3d.utility.Vector3dVector(points.transpose()),
            'color': None
        })

    if point_colors_list:
        colors = np.array(point_colors_list[0])
        # colors = np.full((len(points_list[0][0]), 3), point_colors_list[0])
        for i in range(1, len(points_list)):
            colors = np.vstack((colors, np.full((len(points_list[i][0]), 3), point_colors_list[i])))

        state_to_points[-1]['color'] = o3d.utility.Vector3dVector(colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = state_to_points[-1]['points']

    if point_colors_list:
        pcd.colors = state_to_points[-1]['color']

    points = np.asarray(pcd.points).transpose()

    axes_line_set = gen_axes_lines(points, interval)

    mark_lines_set = gen_lines(points, interval)

    labels = gen_mark_text(points, interval)

    vis = o3d.visualization.VisualizerWithKeyCallback()

    render_opt = o3d.visualization.RenderOption()
    render_opt.mesh_show_back_face = True

    vis.create_window(window_name=name, width=800, height=600)

    vis.add_geometry(const_pcd)
    vis.add_geometry(pcd)
    vis.add_geometry(axes_line_set)
    vis.add_geometry(mark_lines_set)

    for label in labels:
        vis.add_geometry(label)

    vis.get_render_option().mesh_show_back_face = True

    vis.poll_events()
    vis.update_renderer()

    def update_pcd(state):
        print(f'state = {state}, количество точек {len(state_to_points[state]['points'])}')
        pcd.points = o3d.utility.Vector3dVector(state_to_points[state]['points'])
        if point_colors_list is not None:
            pcd.colors = state_to_points[state]['color']

    def handle_key_j(vis):
        global state
        state -= 1
        if state < 0:
            state = len(points_list) - 1
        update_pcd(state)
        return True

    def handle_key_k(vis):
        global state
        state += 1
        if state > len(points_list) - 1:
            state = 0
        update_pcd(state)
        return True

    def handle_key_l(vis):
        global state
        state = -1
        update_pcd(state)
        return True

    vis.register_key_callback(ord("J"), handle_key_j)
    vis.register_key_callback(ord("K"), handle_key_k)
    vis.register_key_callback(ord("L"), handle_key_l)

    print(f'J - назад')
    print(f'K - вперед')
    print(f'L - сброс (отображает все точки)')

    vis.run()
    vis.destroy_window()


def equalize_histogram(image):
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    cdf = hist.cumsum()

    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    equalized_image = equalized_image.reshape(image.shape).astype(np.uint8)

    return equalized_image

def normalize(channel):
    return channel / 256

def main():
    # las_file_path = f'D:\\lases\\work\\pp15\\pp15-denoise-ground.las'
    las_file_path = f'D:\\lases\\work\\pp15-denoise-ground\\pp15-denoise-ground-f.las'
    las_file_path = f'D:\\lases\\work\\pp1.las'
    # las_file_path = f'D:\\lases\\data\\data\\pp{pp}\\data\\pp{pp}-vls-spring.las'
    points_list = []
    interval = 5
    directory = f'C:\\Users\\lalay\\OneDrive\\Документы\\GitHub\\NLSgitdd\\pp20-120-center-0.5-50_FSCT_output'
    file_paths = []
    for item in os.listdir(directory):
        if item.lower().endswith('.las'):
            full_path = os.path.join(directory, item)
            if os.path.isfile(full_path):
                file_paths.append(full_path)
    for las_file_path in file_paths:
        las = laspy.read(las_file_path)
        points = las_to_points(las)

        return_number = np.array(las.return_number)
        number_of_returns = np.array(las.number_of_returns)
        mask = (number_of_returns > 0) & (return_number > 0)
        points = points.T
        points = points[mask]
        points = points.T

        points_list.append(points)


    # def normalize(channel):
    #     channel_min = np.min(channel)
    #     channel_max = np.max(channel)
    #     return ((channel - channel_min) / (channel_max - channel_min) * 255).astype(np.float32)

        red = normalize(las.red)[mask]
        green = normalize(las.green)[mask]
        blue = normalize(las.blue)[mask]

        # red = las.red / max_val
        # green = las.red / max_val
        # blue = las.red / max_val

        # red = exposure.equalize_hist(red)
        # green = exposure.equalize_hist(green)
        # blue = exposure.equalize_hist(blue)

        red /= 256
        green /= 256
        blue /= 256

        colors = [
            (red[i], green[i], blue[i]) for i in range(len(points[0]))
        ]
        colors = np.array(colors, dtype=np.float32)
        #gray = np.dot(colors * 256, [0.299, 0.587, 0.114])

        #gray = equalize_histogram(gray)

        # colors = np.column_stack([gray, gray, gray]) / 256
        print(colors)
        print(len(colors))

    view('cloud',
         points_list,
         interval,
         [colors]
         )


if __name__ == '__main__':
    main()