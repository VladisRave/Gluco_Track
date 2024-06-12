import pandas as pd
import os

def filter_dataframes(input_directory: str, output_directory: str, columns_to_drop_file: str) -> None:
    """
    Фильтрует датафреймы в исходной директории, удаляя указанные колонки, и сохраняет их в новой директории.

    Параметры:
    ----------
    input_directory (str): Путь к исходной директории с датафреймами.
    output_directory (str): Путь к директории для сохранения отфильтрованных датафреймов.
    columns_to_drop_file (str): Путь к CSV файлу с названиями колонок для удаления.

    Возвращает:
    ----------
    None.
    """
    # Загрузка CSV файла с названиями колонок для удаления с указанием кодировки
    with open(columns_to_drop_file, 'r', encoding='windows-1251') as file:
        columns_to_drop = file.readline().strip().split(',')

    # Убедимся, что директория для сохранения существует
    os.makedirs(output_directory, exist_ok=True)

    # Получение списка файлов в исходной директории
    file_list = os.listdir(input_directory)

    for file_name in file_list:
        if file_name.endswith('.csv'):
            # Загрузка датафрейма
            df = pd.read_csv(os.path.join(input_directory, file_name))

            # Удаление указанных колонок
            df_filtered = df.drop(columns=columns_to_drop, errors='ignore')

            # Сохранение отфильтрованного датафрейма в новую директорию
            df_filtered.to_csv(os.path.join(output_directory, file_name), index=False)

# Фильтрация датафреймов
filter_dataframes(
    "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframe_interpolated",
    "C:/Users/User/PycharmProjects/Gluco_Track/3_Data_Science/clean_dataframes",
    "C:/Users/User/PycharmProjects/Gluco_Track/2_Data_Mining/analysis_results/result_drop_columns.csv"
)
