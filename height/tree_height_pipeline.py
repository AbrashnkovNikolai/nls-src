import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tree_height_functions import (
    process_segment_fast,
    build_chm_with_stratification
)


def fast_height_pipeline(segments, centers_dict, all_points, output_path):
    """
    Ускоренный пайплайн определения высот деревьев

    Args:
        segments: Словарь сегментов {seg_id: points_array}
        centers_dict: Словарь центров {seg_id: (x, y, radius)}
        all_points: Все точки облака
        output_path: Путь для сохранения результатов
    """
    print(f"Запуск пайплайна обработки {len(segments)} сегментов...")

    # Преобразуем в numpy array один раз
    points_array = np.array(all_points, dtype=np.float32)

    # Строим CHM (опционально, для дополнительного анализа)
    print("Построение CHM...")
    chm, min_x, min_y, max_x, max_y = build_chm_with_stratification(
        points_array,
        pixel_size=1.0,
        height_thresholds=[1.0, 3.0, 8.0, 15.0]
    )
    print(f"CHM построен: {chm.shape[1]} x {chm.shape[0]} пикселей")

    # Параллельная обработка сегментов
    print("Обработка сегментов...")
    results = Parallel(n_jobs=2)(
        delayed(process_segment_fast)(
            seg_id, segment_points, centers_dict, points_array
        )
        for seg_id, segment_points in segments.items()
    )

    # Создаем и сохраняем результаты
    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_path, sep=';', index=False)
        print(f"✅ Успешно обработано {len(final_df)} деревьев")
        print(f"📁 Результаты сохранены в: {output_path}")

        # Базовая статистика
        if len(final_df) > 0:
            print(f"📊 Статистика высот:")
            print(f"   Мин. высота: {final_df['h'].min():.2f} м")
            print(f"   Макс. высота: {final_df['h'].max():.2f} м")
            print(f"   Сред. высота: {final_df['h'].mean():.2f} м")
            print(f"   Мед. высота: {final_df['h'].median():.2f} м")

        return final_df
    else:
        print("❌ Не удалось обработать деревья")
        return pd.DataFrame()


def main():
    """
    Пример использования пайплайна
    """
    # Здесь должна быть ваша логика загрузки данных
    # segments = ...
    # centers_dict = ...
    # all_points = ...
    # output_path = "result_trees.csv"

    # Запуск пайплайна
    # results = fast_height_pipeline(segments, centers_dict, all_points, output_path)
    pass


if __name__ == "__main__":
    main()