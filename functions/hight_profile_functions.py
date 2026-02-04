
from imports import *
test = Polygon([
    (357430.910, 6758474.922),  # left
    (357462.474, 6758520.326),  # top
    (357550.327, 6758456.846),  # right
    (357512.8682, 6758413.7518)  # bot
])


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def build_rectangle_from_points(point1, point2, d):
    """
    Строит прямоугольник, в котором данные точки лежат на центрах противоположных сторон,
    а сами стороны равны 2d.

    Parameters:
    point1, point2: tuple (x, y) - координаты двух точек
    d: float - половина длины стороны прямоугольника

    Returns:
    vertices: list of tuples - вершины прямоугольника в порядке обхода
    """
    x1, y1 = point1
    x2, y2 = point2

    # Вектор между точками
    dx = x2 - x1
    dy = y2 - y1

    # Нормализованный перпендикулярный вектор
    length = np.sqrt(dx ** 2 + dy ** 2)
    if length == 0:
        raise ValueError("Точки не могут совпадать")

    # Перпендикулярный вектор (поворот на 90 градусов)
    perp_dx = -dy / length
    perp_dy = dx / length

    # Вычисляем вершины прямоугольника
    # Вершины для стороны, содержащей point1
    v1 = (x1 + perp_dx * d, y1 + perp_dy * d)
    v2 = (x1 - perp_dx * d, y1 - perp_dy * d)

    # Вершины для стороны, содержащей point2
    v3 = (x2 - perp_dx * d, y2 - perp_dy * d)
    v4 = (x2 + perp_dx * d, y2 + perp_dy * d)

    return [v1, v2, v3, v4]


# Пример использования
def plot_rectangle(point1, point2, d):
    """Визуализация прямоугольника"""
    vertices = build_rectangle_from_points(point1, point2, d)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Рисуем прямоугольник
    rect = patches.Polygon(vertices, closed=True, alpha=0.3, color='blue')
    ax.add_patch(rect)

    # Рисуем исходные точки
    ax.plot(point1[0], point1[1], 'ro', markersize=8, label=f'Точка 1: {point1}')
    ax.plot(point2[0], point2[1], 'go', markersize=8, label=f'Точка 2: {point2}')

    # Показываем центры сторон
    side1_center = ((vertices[0][0] + vertices[1][0]) / 2, (vertices[0][1] + vertices[1][1]) / 2)
    side2_center = ((vertices[2][0] + vertices[3][0]) / 2, (vertices[2][1] + vertices[3][1]) / 2)

    ax.plot(side1_center[0], side1_center[1], 'bo', markersize=6)
    ax.plot(side2_center[0], side2_center[1], 'bo', markersize=6)

    # Настройки графика
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Прямоугольник с d = {d}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    return vertices



def generate_points_on_line(point1, point2, step=0.5):
    """
    Генерирует точки на прямой между двумя точками с заданным шагом.

    Parameters:
    point1, point2: tuple (x, y) - координаты двух точек
    step: float - шаг между точками

    Returns:
    points: list of tuples - точки на прямой между point1 и point2
    """
    x1, y1 = point1
    x2, y2 = point2

    # Вектор направления
    dx = x2 - x1
    dy = y2 - y1

    # Длина отрезка
    length = np.sqrt(dx ** 2 + dy ** 2)

    if length == 0:
        return [point1]  # Точки совпадают

    # Нормализованный вектор направления
    unit_dx = dx / length
    unit_dy = dy / length

    # Генерируем точки
    points = []
    current_distance = 0

    # Добавляем точки пока не достигнем конца отрезка
    while current_distance <= length:
        x = x1 + unit_dx * current_distance
        y = y1 + unit_dy * current_distance
        points.append((x, y))
        current_distance += step

    # Гарантируем, что последняя точка точно совпадает с point2
    if points[-1] != point2:
        points.append(point2)

    return points


# Альтернативная версия с фиксированным количеством точек
def generate_points_on_line_fixed_count(point1, point2, num_points=10):
    """
    Генерирует фиксированное количество точек на прямой между двумя точками.

    Parameters:
    point1, point2: tuple (x, y) - координаты двух точек
    num_points: int - количество точек (включая концы)

    Returns:
    points: list of tuples - точки на прямой между point1 и point2
    """
    x1, y1 = point1
    x2, y2 = point2

    points = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        points.append((x, y))

    return points


# Визуализация
def plot_points_on_line(point1, point2, step=0.5):
    """Визуализация точек на прямой"""
    points = generate_points_on_line(point1, point2, step)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Рисуем исходные точки
    ax.plot(point1[0], point1[1], 'ro', markersize=10, label=f'Точка 1: {point1}')
    ax.plot(point2[0], point2[1], 'go', markersize=10, label=f'Точка 2: {point2}')

    # Рисуем сгенерированные точки
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    ax.plot(x_vals, y_vals, 'b-', alpha=0.5, linewidth=2)
    ax.plot(x_vals, y_vals, 'bo', markersize=4, alpha=0.7, label=f'Точки с шагом {step}')

    # Подписываем некоторые точки
    for i, point in enumerate(points[::5]):  # Каждую 5-ю точку
        ax.annotate(f'({point[0]:.1f}, {point[1]:.1f})',
                    xy=point, xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Точки на прямой с шагом {step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    return points



def get_paralel_segment(border,d,is_fixed_count,count):
    test = [
        (357430.910, 6758474.922),  # left
        (357462.474, 6758520.326),  # top
        (357550.327, 6758456.846),  # right
        (357512.8682, 6758413.7518)  # bot
    ]
    border = test

    upper_length = (border[0], border[2])

    bot_length = (border[1], border[3])
    upper_points = generate_points_on_line(upper_length[0], upper_length[1], step=3)
    bot_points = generate_points_on_line(bot_length[0], bot_length[1], step=3)
    lines =[]
    for i in len(upper_points):
        line = (upper_points[i],bot_points[i])
        lines.append(line)

    segments = [build_rectangle_from_points(line[0],line[1],3) for line in lines]
    return segments

# Тестирование
if __name__ == "__main__":
    # Пример 1
    point1 = (1, 1)
    point2 = (4, 3)
    d = 1.5

    vertices = plot_rectangle(point1, point2, d)
    print("Вершины прямоугольника:")
    for i, vertex in enumerate(vertices):
        print(f"Вершина {i + 1}: ({vertex[0]:.2f}, {vertex[1]:.2f})")

    # Пример 2 - горизонтальный прямоугольник
    point1 = (0, 0)
    point2 = (5, 0)
    d = 1

    vertices = plot_rectangle(point1, point2, d)
    print("\nВершины горизонтального прямоугольника:")
    for i, vertex in enumerate(vertices):
        print(f"Вершина {i + 1}: ({vertex[0]:.2f}, {vertex[1]:.2f})")
    points = [point1,point2]

    # Дополнительный пример с другими точками
    print("\n" + "=" * 50)
    print("Дополнительный пример:")

    point1 = (0, 0)
    point2 = (5, 5)
    step = 1.0

    points_simple = generate_points_on_line(point1, point2, step)
    print(f"Точки от {point1} до {point2} с шагом {step}:")
    for i, point in enumerate(points_simple):
        print(f"{i + 1:2d}. ({point[0]:.1f}, {point[1]:.1f})")