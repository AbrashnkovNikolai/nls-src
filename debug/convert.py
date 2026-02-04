from pathlib import Path
from functions.las_functions import shp_to_csv
# Укажите путь к вашей директории
DIR_PATH = "csv_etalons/etalons"

# Конвертируем все .shp файлы
shp_file = Path('etalons/pp54-res.shp')
csv_file = shp_file.with_suffix('.csv')
print(f"Конвертация: {shp_file.name}")
shp_to_csv(str(shp_file), str(csv_file))