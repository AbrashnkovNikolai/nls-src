import pandas as pd
import matplotlib.pyplot as plt

def plot_h_distribution(csv_file='pp57_progon_chm.csv'):
    """
    Рисует распределение столбца 'h' из CSV файла

    Parameters:
    csv_file (str): путь к CSV файлу (по умолчанию 'testovi.csv')
    """
    try:
        # Чтение данных
        df = pd.read_csv(csv_file,sep=';')

        # Проверка наличия столбца 'h'
        if 'h' not in df.columns:
            print(f"Ошибка: столбец 'h' не найден в файле {csv_file}")
            print(f"Доступные столбцы: {list(df.columns)}")
            return

        # Создание графика
        plt.figure(figsize=(10, 6))

        # Гистограмма распределения
        plt.subplot(1, 2, 1)
        plt.hist(df['h'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Гистограмма распределения h')
        plt.xlabel('Значения h')
        plt.ylabel('Частота')


        plt.tight_layout()
        plt.show()


    except FileNotFoundError:
        print(f"Ошибка: файл {csv_file} не найден")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


# Использование функции
plot_h_distribution('testovi.csv')