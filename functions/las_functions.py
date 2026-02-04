
from typing import Any

import laspy
from laspy import LasData
import pandas as pd
import numpy as np
from numpy import dtype
from shapely import Point
import geopandas as gpd

#from logger import logger

def csv_to_points(
    csv_file_path: str,
    x_column: str = 'x',
    y_column: str = 'y',
    #z_column: str = 'z',
    sep = ';',
    indent: int = 2
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
    try:
        # Читаем CSV файл
        df = pd.read_csv(csv_file_path,sep=sep)

        # Проверяем наличие необходимых колонок
        #z_column
        required_columns = [x_column, y_column, ]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Отсутствуют необходимые колонки: {missing_columns}")

        # Извлекаем координаты
        x = np.array(df[x_column], dtype=np.float64)
        y = np.array(df[y_column], dtype=np.float64)
        #z = np.array(df[z_column], dtype=np.float64)


        return np.vstack((x, y))
    finally:
        print('ксв преобразован в точки')



def las_to_points(
        las: LasData,
        indent: int = 2
) -> np.ndarray[Any, dtype[Any]]:
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)

    #logger.info(f'{" " * indent}Количество точек = {len(x):_}')

    return np.vstack((x, y, z), dtype=np.float64)


def write_points_to_las(
        points,
        out_file_path: str,
        point_format,
        file_version: str,
        extra: dict = None
) -> None:
    new_las = laspy.create(
        point_format=point_format,
        file_version=file_version
    )

    new_las.x = points[0]
    new_las.y = points[1]
    new_las.z = points[2]

    if extra:
        for attr_name, attr_values in extra.items():
            if hasattr(new_las, attr_name):
                setattr(new_las, attr_name, attr_values)
            else:
                new_las.add_extra_dim(laspy.ExtraBytesParams(
                        name=attr_name,
                        type=type(attr_values[0]),
                        description=f"Extra dimension {attr_name}"
                    )
                )
                setattr(new_las, attr_name, attr_values)

    new_las.write(out_file_path)


def shp_to_las(
        shp_file_path: str,
        las_file_path: str
) -> None:
    shp = gpd.read_file(shp_file_path)
    print(list(shp.h))
    z = np.array(list(shp.h))
    z = np.nan_to_num(z, nan=-1)
    print(z.min(), z.max())
    x = []
    y = []
    for p in shp.geometry:
        x_coords = [x for x, y in p.exterior.coords]
        y_coords = [y for x, y in p.exterior.coords]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        x.append(center_x)
        y.append(center_y)

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    points = np.vstack((x, y, z), dtype=np.float64)

    las = laspy.read(f'D:\\lases\\work\\pp15-denoise-ground.las')

    write_points_to_las(points, las_file_path, las.point_format, str(las.header.version))

    return
def csv_to_shp(csv_path,shp_path):
# Загрузка CSV-файла
    df = pd.read_csv(csv_path)

    # Создание геометрии для точечных данных
    geometry = [Point(x,y).buffer(d/2) for x,y,d in zip(df['x_tree_base'], df['y_tree_base'],df['DBH'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:32639")

    # Сохранение в шейп-файл
    gdf.to_file(shp_path)
#csv_to_shp('pp20-120-center-0.5-50_FSCT_output/tree_data.csv','FSCT_bigger.shp')
def shp_to_points(shp_file_path):
    shp = gpd.read_file(shp_file_path)
    #print(list(shp.h))
    #z = np.array(list(shp.Height))
    #z = np.nan_to_num(z, nan=-1)
    #print(z.min(), z.max())
    x = []
    y = []
    for p in shp.geometry:
        x_coords = [x for x, y in p.exterior.coords]
        y_coords = [y for x, y in p.exterior.coords]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)

        x.append(center_x)
        y.append(center_y)

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    points = np.vstack((x, y), dtype=np.float64)
    return points

def get_sliced_points(las_path, min_h: float, max_h: float):
        """чтение и получение среза с лас файла"""
        print(f"Фильтрация точек {min_h}-{max_h} м...")
        points_list = []
        min_z = 0
        # Второй проход: применяем нормализацию
        with laspy.open(las_path) as las:
            for chunk in las.chunk_iterator(50_000_000):
                mask = (chunk.z >= min_h) & (chunk.z <= max_h)
                if np.any(mask):
                    points = np.column_stack((chunk.x[mask], chunk.y[mask], chunk.z[mask]))
                    points_list.append(points)
        if not points_list:
            raise ValueError(f"Нет точек в диапазоне {min_h}-{max_h} м")
        sliced_points = np.vstack(points_list)
        #sliced_points = sliced_points.transpose()
        print(f"Найдено {len(sliced_points):,} точек в слое")
        return sliced_points


def shp_to_csv(
        shp_file_path: str,
        csv_file_path: str,
        geometry_to_coords: bool = True,
        coord_columns: list[str] = None,
        sep: str = ';',
        encoding: str = 'utf-8'
) -> None:
    """
    Конвертирует shapefile в CSV файл.

    Args:
        shp_file_path: Путь к входному shapefile
        csv_file_path: Путь для сохранения CSV файла
        geometry_to_coords: Если True, преобразует геометрию в координатные столбцы
        coord_columns: Список имен столбцов для координат (по умолчанию ['x', 'y'])
        sep: Разделитель для CSV файла
        encoding: Кодировка файла

    Returns:
        None
    """
    try:
        # Читаем shapefile
        gdf = gpd.read_file(shp_file_path)

        # Если нужно преобразовать геометрию в координаты
        if geometry_to_coords:
            if coord_columns is None:
                coord_columns = ['x', 'y']

            # Извлекаем координаты из геометрии
            if gdf.geom_type.iloc[0] == 'Point':
                # Для точечных объектов
                gdf[coord_columns[0]] = gdf.geometry.x
                gdf[coord_columns[1]] = gdf.geometry.y
            else:
                # Для полигонов и других типов - используем центроиды
                gdf[coord_columns[0]] = gdf.geometry.centroid.x
                gdf[coord_columns[1]] = gdf.geometry.centroid.y

            # Удаляем столбец геометрии
            gdf = gdf.drop(columns=['geometry'])

        # Сохраняем в CSV
        gdf.to_csv(csv_file_path, sep=sep, encoding=encoding, index=False)

        print(f"Shapefile успешно конвертирован в CSV: {csv_file_path}")
        print(f"Количество строк: {len(gdf)}")
        print(f"Колонки: {list(gdf.columns)}")

    except Exception as e:
        print(f"Ошибка при конвертации shapefile в CSV: {e}")
        raise