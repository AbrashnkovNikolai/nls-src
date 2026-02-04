import math
import random
from typing import List, Tuple, Optional

Point = Tuple[float, float]


class RobustCircleDetector:
    '''класс для нахождения окружности покрывающий % точек кластера'''
    def __init__(self, points: Optional[List[Point]] = None):
        self.points = list(points) if points is not None else []
        self.center: Optional[Point] = None
        self.radius: Optional[float] = None
        self.coverage: Optional[float] = None

    def _validate_point(self, p: Point) -> bool:
        """Проверяет, что точка корректна (не None и содержит 2 координаты)."""
        return p is not None and len(p) == 2 and all(isinstance(x, (int, float)) for x in p)

    def add_point(self, point: Point) -> None:
        if self._validate_point(point):
            self.points.append(point)
            self._reset()
    def clear_points(self) -> None:
        self.points.clear()
        self._reset()

    def add_points(self, points: List[Point]) -> None:
        valid_points = [p for p in points if self._validate_point(p)]
        self.points.extend(valid_points)
        self._reset()

    def _reset(self) -> None:
        self.center = None
        self.radius = None
        self.coverage = None

    @staticmethod
    def _distance(p1: Point, p2: Point) -> float:
        try:
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        except (TypeError, IndexError):
            return float('inf')

    def _make_circle(self, points: List[Point]) -> Tuple[Optional[Point], Optional[float]]:
        """Возвращает (center, radius) или (None, None) при ошибке."""
        if not points:
            return None, None

        try:
            # Фильтруем некорректные точки
            valid_points = [p for p in points if self._validate_point(p)]
            if not valid_points:
                return None, None

            shuffled = valid_points.copy()
            random.shuffle(shuffled)

            center = shuffled[0]
            radius = 0.0

            for i in range(1, len(shuffled)):
                dist = self._distance(center, shuffled[i])
                if dist <= radius:
                    continue

                center = shuffled[i]
                radius = 0.0

                for j in range(i):
                    dist = self._distance(center, shuffled[j])
                    if dist <= radius:
                        continue

                    center = (
                        (shuffled[i][0] + shuffled[j][0]) / 2,
                        (shuffled[i][1] + shuffled[j][1]) / 2
                    )
                    radius = self._distance(center, shuffled[i])

                    for k in range(j):
                        dist = self._distance(center, shuffled[k])
                        if dist <= radius:
                            continue

                        # Вычисляем окружность по трем точкам
                        try:
                            center = self._circumcenter(shuffled[i], shuffled[j], shuffled[k])
                            radius = self._distance(center, shuffled[i])
                        except:
                            continue

            return center, radius
        except:
            return None, None

    def compute(self) -> bool:
        """Вычисляет окружность. Возвращает True при успехе."""
        self.center, self.radius = self._make_circle(self.points)
        self.coverage = 1.0 if self.center is not None else 0.0
        return self.center is not None

    def compute_robust(self, coverage: float = 0.98, max_iter: int = 50) -> bool:
        if not self.points:
            self.center = None
            self.radius = None
            return False

        if float(coverage) >= 1.0:
            return self.compute()

        target_count = max(1, int(len(self.points) * coverage))
        best_center, best_radius = None, float('inf')

        for _ in range(max_iter):
            # Выбираем случайные точки для инициализации
            sample = random.sample(self.points, min(3, len(self.points)))
            center, radius = self._make_circle(sample)

            if center is None:
                continue

            # Считаем расстояния и находим радиус для coverage
            distances = [self._distance(center, p) for p in self.points]
            distances.sort()
            radius_candidate = distances[target_count - 1]

            if radius_candidate < best_radius:
                best_center, best_radius = center, radius_candidate

        self.center, self.radius = best_center, best_radius
        self.coverage = coverage if best_center is not None else 0.0
        return self.center is not None

    def get_circle(self) -> Tuple[Optional[Point], Optional[float]]:
        """Безопасный метод получения окружности."""
        if self.center is None or self.radius is None:
            if not self.compute():
                return None, None
        return self.center, self.radius

    def get_outliers(self, threshold: float = 1.0) -> List[Point]:
        if self.center is None or self.radius is None:
            return []
        return [
            p for p in self.points
            if self._distance(p, self.center) > self.radius * threshold
        ]


# Пример использования
''' 
    test_data = [
        (0, 0), (1, 1), (2, 2), None,
        ("text", 5), (3, 3), (float('nan'), 0)
    ]

    detector = RobustCircleDetector()
    detector.add_points(test_data)

    if detector.compute_robust(coverage=0.9):
        center, radius = detector.get_circle()
        print(f"Center: {center}, Radius: {radius}")
        print(f"Outliers: {detector.get_outliers()}")
    else:
        print("Не удалось вычислить окружность для данных точек")'''
