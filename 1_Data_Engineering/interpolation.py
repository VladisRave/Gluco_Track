import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

def interpolate_files(input_dir: str, output_dir: str, points: int = 3006) -> None:
    """
    Интерполирует файлы, содержащие 5001 точку, до указанного количества точек.

    Параметры:
    input_dir (str): Путь к директории с исходными файлами.
    output_dir (str): Путь к директории для сохранения интерполированных файлов.
    points (int): Количество точек, до которого будет произведена интерполяция. По умолчанию 3006.
    """

    # Получение списка файлов в директории
    files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # Создание объекта tqdm для отображения прогресса
    progress_bar = tqdm(files, desc="Интерполяция файлов", unit="файл")

    # Перебор каждого файла
    for file in progress_bar:
        # Обновление прогресс-бара
        progress_bar.set_postfix(current_file=file)

        # Чтение DataFrame из файла
        df = pd.read_csv(os.path.join(input_dir, file))

        # Создание функций интерполяции для каждого столбца
        interpolated_functions = {col: interp1d(df.index, df[col], kind='cubic') for col in df.columns}

        # Создание нового индекса с меньшим количеством точек
        new_index = np.linspace(df.index.min(), df.index.max(), points)

        # Интерполяция значений для каждого столбца
        interpolated_data = {col: func(new_index) for col, func in interpolated_functions.items()}

        # Создание нового DataFrame с интерполированными значениями
        df_interpolated = pd.DataFrame(interpolated_data, index=new_index)

        # Извлечение имени файла из пути
        file_name = os.path.basename(file)

        # Сохранение DataFrame в новом файле CSV
        new_path = os.path.join(output_dir, "interpolated_" + file_name)
        df_interpolated.to_csv(new_path, index=False)

    print("Процесс интерполяции файлов завершён!")

# Пример использования функции
input_dir = "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes"
output_dir = "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframe_interpolated"
interpolate_files(input_dir, output_dir)
