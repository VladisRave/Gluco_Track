import os
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

def standardize_data(df: pd.DataFrame, target_variable: str) -> pd.DataFrame:
    """
    Стандартизация данных.

    Параметры:
    ----------
    df (pd.DataFrame): Входной DataFrame для стандартизации.
    target_variable (str): Название целевой переменной в DataFrame.

    Возвращает:
    ----------
    pd.DataFrame-Стандартизованный DataFrame.
    """
    y = df[target_variable]
    X = df.drop(columns=[target_variable])

    X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_standardized = X_standardized / np.sqrt(np.sum(X_standardized ** 2, axis=0))

    scaled_df = deepcopy(X_standardized)
    scaled_df[target_variable] = y.values

    return scaled_df

def plot_random_heatmaps_from_directory(input_directory:str, target_column:str, num_heatmaps:int=5)->None:
    """
        Фильтрует датафреймы в исходной директории, удаляя указанные колонки, и сохраняет их в новой директории.

        Параметры:
        ----------
        input_directory (str): Путь к исходной директории с датафреймами.
        target_column (str): Название целевой переменной, которую нужно исключить.
        num_heatmaps (int): Количество тепловых карт для построения (по умолчанию 5).

        Возвращает:
        ----------
        None.
    """
    # Получение списка файлов с расширением .csv в указанной директории
    file_list = [file for file in os.listdir(input_directory) if file.endswith('.csv')]

    # Формирование полного пути к каждому файлу
    full_name_list = [os.path.abspath(os.path.join(input_directory, file)) for file in file_list]

    # Проверка, что есть достаточно файлов для выбора
    if len(file_list) < num_heatmaps:
        print(f"Недостаточно файлов в директории для создания {num_heatmaps} тепловых карт.")
        num_heatmaps = len(file_list)

    # Выбор случайных файлов
    random_files = random.sample(full_name_list, num_heatmaps)

    for file_name in random_files:

        # Открываем существующий файл для чтения
        df = pd.read_csv(file_name)
        # Переименовываем столбцы, чтобы они были валидными идентификаторами
        df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")

        # Выделяем X и стандартизуем его
        labels = df.drop(columns=["Концентрация_глюкозы"]).columns.tolist()
        df = standardize_data(df, "Концентрация_глюкозы")

        # Получение списка колонок, исключая целевую переменную
        feature_columns = [col for col in df.columns if col != target_column]

        if len(feature_columns) < 2:
            print(f"Недостаточно колонок для создания тепловой карты в файле {file_name}")
            continue

        # Выбор колонок для матрицы корреляции
        correlation_matrix = df[labels].corr()

        # Построение тепловой карты
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        # Получение имени файла из полного пути
        file_name_with_extension = os.path.basename(file_name)
        # Отделение имени файла от его расширения
        name_file, file_extension = os.path.splitext(file_name_with_extension)
        print(name_file)
        print(file_extension)
        plt.title(f'Тепловая карта файла: {name_file}')
        plt.show()

# Построение случайных тепловых карт
plot_random_heatmaps_from_directory(
                                    "C:/Users/User/PycharmProjects/Gluco_Track/3_Data_Science/clean_dataframes",
                                    'Концентрация глюкозы',
                                    1
                                    )