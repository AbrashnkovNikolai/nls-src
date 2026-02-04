import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

import laspy

from tree_detection.las_functions import las_to_points, write_points_to_las
from tree_detection.normalization import normalize_height_knnidw
from tree_detection.filters import filter_ground, filter_noise_z, filter_sor_xy
from tree_detection.rasterization import *


def main():

    #las_file = f'D:/lases/work/vls/vls-cut.las'
    # las_file = f'D:/lases/work/vls-cut-res/vls-cut-f-3.las'
    # las_file = f'D:/lases/work/vls-cut-res/vls-cut-without-noise-ground.las'

    las_file = f'pp20cutbuf5NORM-norm-nograss.las'
    # las_file = f'D:/lases/work/pp57-raw-cut-res/pp57-raw-cut-without-noise-ground.las'

    las = laspy.read(las_file)
    points = las_to_points(las)

    points = filter_noise_z(points)
    points, ground = filter_ground(points)
    #points = normalize_height_knnidw(points, ground)
    # points = filter_sor_xy(points, 50, 0.7)

    '''write_points_to_las(
        points,
        'pp20filtered',
        las.point_format,
        las.header.version
    )
'''
    points = points.T

    z_step = 0.3
    pixel_size = 0.2

    raster, min_x, min_y, max_x, max_y = build_raster_chunked_discrete(points, z_step, pixel_size, 1)
    raster = normalize(raster)

    count_raster, min_x, min_y, max_x, max_y = build_raster_chunked_point_count(points, pixel_size, 2)
    count_raster = normalize(count_raster)

    # max_z_raster, min_x, min_y, max_x, max_y = build_max_z_raster(points, pixel_size)
    # norm_max_z_raster = normalize(max_z_raster)

    # chm_raster, min_x, min_y, max_x, max_y = build_chm(points, pixel_size, True, 0)
    chm_raster, min_x, min_y, max_x, max_y = build_chm_with_stratification(
        points, pixel_size, [10.0, 16.0, 22.0, 28.0]
    )
    norm_chm_raster = normalize(chm_raster)

    variance_raster, min_x, min_y, max_x, max_y = build_raster_height_variance(points, pixel_size, 1)
    variance_raster = normalize(variance_raster)

    mk = [0.4, 0.3, 0.3, 0]

    raster = raster * mk[0] + count_raster * mk[1] + variance_raster * mk[2] + norm_chm_raster * mk[3]
    raster = normalize(raster)

    raster[raster < 0.3] = 0
    maxima_mask = find_local_maxima_mask(raster, 5, 0.35)

    res = maxima_mask_to_points(maxima_mask, chm_raster, min_x, min_y, pixel_size)




    print(len(res))
    res = res[(res[:, 2]) > 6]
    print(len(res))

    #write_points_to_las(
    #    res.T, 'D:/lases/work/raster/test-raster-res.las', las.point_format, las.header.version
    #)

    save_raster_to_tiff(
        chm_raster,
        'pp20-raster.tiff',
        min_x,
        min_y,
        pixel_size
    )

    empty_mask = count_raster <= 0.000001
    count_raster = count_raster.astype(float)
    count_raster[empty_mask] = np.nan

    save_raster_to_tiff(
        maxima_mask,
        'pp20-maxima-mask-raster.tiff',
        min_x,
        min_y,
        pixel_size
    )

    save_raster_to_tiff(
        count_raster,
        'pp20-count-raster.tiff',
        min_x,
        min_y,
        pixel_size
    )

    empty_mask = variance_raster <= 0.000001
    variance_raster = variance_raster.astype(float)
    variance_raster[empty_mask] = np.nan

    save_raster_to_tiff(
        variance_raster,
        'pp20-variance-raster.tiff',
        min_x,
        min_y,
        pixel_size
    )

    max_z_raster = raster <= 0.000001
    max_z_raster = max_z_raster.astype(float)
    max_z_raster[empty_mask] = np.nan

    save_raster_to_tiff(
        max_z_raster,
        'pp20-max-z-raster.tiff',
        min_x,
        min_y,
        pixel_size
    )

    empty_mask = raster <= 0.000001
    raster = raster.astype(float)
    raster[empty_mask] = np.nan

    save_raster_to_tiff(
        raster,
        'pp20-last-raster.tiff',
        min_x,
        min_y,
        pixel_size
    )

    visualize_raster_white_background(raster, res, min_x, min_y)


if __name__ == "__main__":
    main()
