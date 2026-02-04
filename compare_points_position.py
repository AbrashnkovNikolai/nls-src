import laspy
from functions.las_functions import las_to_points
import numpy as np


def compare(etalon_file_path,res_file_path):
    #etalon_file_path = f'D:\\lases\\work\\pp15-120-res.las'
    #res_file_path = f'D:\\lases\\work\\pp15-denoise-ground\\pp15-denoise-ground-lm-1.las'
    # res_file_path = f'D:\\lases\\work\\pp15-denoise-ground-best-auto\\pp15-denoise-ground-res.las'

    las = laspy.read(etalon_file_path)
    etalon_points = las_to_points(las, indent=0)

    las = laspy.read(res_file_path)
    res_points = las_to_points(las, indent=0)

    etalon_points = etalon_points.transpose()
    res_points = res_points.transpose()

    etalon_points = etalon_points[etalon_points[:, 2] > 6]
    res_points = res_points[res_points[:, 2] > 6]

    etalon_points = etalon_points[etalon_points[:, 2].argsort()[::-1]]
    res_points = res_points[res_points[:, 2].argsort()[::-1]]

    max_l = 2.23  # np.sqrt(50 ** 2 + 100 ** 2 + 50 ** 2)

    print()
    print(f'Количество точек в эталоне( > 6) = {len(etalon_points)}')
    print(f'Количество точек в решении( > 6) = {len(res_points)}')

    d = {}
    close = []
    for i in range(0, len(etalon_points)):
        ep = etalon_points[i]
        temp_l = max_l
        point = -1
        for j in range(0, len(res_points)):
            rp = res_points[j]
            cl = np.sqrt((rp[0] - ep[0]) ** 2 + (rp[1] - ep[1]) ** 2)  # + (rp[2] - ep[2]) ** 2)
            if j not in close and cl < temp_l:
                point = j
                temp_l = cl

        d.setdefault(i, point)
        # close.append(point)

    ml = []
    k = 0
    for key, value in d.items():
        if value != -1:
            ep = etalon_points[key]
            rp = res_points[value]

            r = np.sqrt((rp[0] - ep[0]) ** 2 + (rp[1] - ep[1]) ** 2)  # + (rp[2] - ep[2]) ** 2)

            ml.append(r)

            # print(f'{key} - {value}, r = {r:.2f}')
            k += 1
        else:
            pass
            # print(f'{key} - {value}, r = -1')

    ml = np.array(ml)
    print()
    print(f'{max_l=}')
    print(f'{k} точек из {len(etalon_points)} нашли пару({k / len(etalon_points) * 100:.2f} %)')
    print()
    print(f'Минимальное расстояние = {ml.min():.2f}')
    print(f'Максимальное расстояние = {ml.max():.2f}')
    print(f'Среднее расстояние = {ml.mean():.2f}')
    print()
    print(f'Количество точек с расстоянием < {ml.mean() * 4:.2f} = {len(ml[ml < ml.mean() * 4])}')
    print(f'Количество точек с расстоянием < {ml.mean() * 2:.2f} = {len(ml[ml < ml.mean() * 2])}')
    print(f'Количество точек с расстоянием < {ml.mean():.2f} = {len(ml[ml < ml.mean()])}')
    print(f'Количество точек с расстоянием < {ml.mean() / 2:.2f} = {len(ml[ml < ml.mean() / 2])}')
    print(f'Количество точек с расстоянием < {ml.mean() / 4:.2f} = {len(ml[ml < ml.mean() / 4])}')


if __name__ == '__main__':
    compare()
