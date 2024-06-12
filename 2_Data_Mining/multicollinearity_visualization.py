import os
import csv
import glob
import pandas as pd
import matplotlib.pyplot as plt


def select_common_elements(position_dict, threshold=0.5)->list:
    """
    Функция выбирает только те элементы листа, которые встречаются в половине или более исследуемых датафреймов.

    Параметры:
    ----------
    position_dict (dict): Словарь с ключами-столбцами и значениями-списками позиций.
    threshold (float): Пороговое значение для определения часто используемых элементов.
                       По умолчанию установлено значение 0.5.

    Возвращает:
    ----------
    selected_elements (list): Список часто используемых элементов.
    """
    total_files = len(position_dict)
    selected_elements = []
    for column, positions in position_dict.items():
        count = len(positions)
        if count / total_files >= threshold:
            selected_elements.append(column)
    return selected_elements


def analyze_removed_columns_selected(input_directory: str, output_dir: str, threshold=0.5) -> None:
    """
    Анализ удалённых столбцов и построение гистограммы их частоты с учетом часто используемых элементов.

    Параметры:
    ----------
    input_dir (str): Путь к директории, содержащей все файлы экспериментов.
    output_dir (str): Путь к выходной директории, содержащей файл отчета.
    threshold (float): Пороговое значение для определения часто используемых элементов.
                       По умолчанию установлено значение 0.5.

    Возвращает:
    ----------
    None
    """
    # Путь к файлу отчета
    report_file_path = os.path.join(output_dir, 'removed_columns_report.csv')

    # Чтение файла
    dataframe_removed_columns = pd.read_csv(report_file_path)

    # Подсчёт количества файлов эксперимента
    num_exper = len(glob.glob(os.path.join(input_directory, '*.csv')))

    # Инициализируем словарь для хранения позиций
    position_dict = {}

    # Обходим строки DataFrame и обновляем словарь позиций
    for index, row in dataframe_removed_columns.iterrows():
        removed_columns_list = eval(row['Список удалённых столбцов'])
        for position, column in enumerate(removed_columns_list, start=1):
            if column not in position_dict:
                position_dict[column] = []
            position_dict[column].append(position)

    # Создание нового рисунка
    plt.figure(figsize=(12, 6))

    # Построение первой гистограммы
    plt.subplot(1, 2, 1)
    plt.bar(position_dict.keys(), [len(positions) / num_exper for positions in position_dict.values()], color='skyblue')

    # Поворот меток на оси X на 90 градусов
    plt.xticks(rotation=90)
    # Добавление заголовка и подписей к осям
    plt.title('Частота всех удаляемых столбцов')
    plt.xlabel('Колонки')
    plt.ylabel('Частота удаления')
    # Добавление сетки
    plt.grid(True)

    # Выбираем часто используемые элементы
    selected_elements = select_common_elements(position_dict, threshold)
    # Сохранение списка в CSV файл
    with open("C:/Users/User/PycharmProjects/Gluco_Track/2_Data_Mining/analysis_results/result_drop_columns.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(selected_elements)

    # Построение гистограммы только для часто используемых элементов
    count_positions_selected = {column: len(positions)/num_exper for column, positions in position_dict.items() if
                                column in selected_elements}

    # Построение второй гистограммы
    plt.subplot(1, 2, 2)
    plt.bar(count_positions_selected.keys(), count_positions_selected.values(), color='skyblue')
    # Поворот меток на оси X на 90 градусов
    plt.xticks(rotation=90)
    # Добавление заголовка и подписей к осям
    plt.title('Частота наиболее удаляемых столбцов')
    plt.xlabel('Колонки')
    plt.ylabel('Количество элементов')
    # Добавление сетки
    plt.grid(True)
    # Отображение гистограммы
    plt.show()

    # Переименовываем столбцы, чтобы они были валидными идентификаторами
    selected_elements = [column.replace('_', ' ') for column in selected_elements]

    # Выводим количество удалённых элементов для каждого столбца
    print("\nЧасто используемые элементы:")
    for column in selected_elements:
        print(f"-{column}")

# Пример использования функций
analyze_removed_columns_selected(
                "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframe_interpolated",
                "C:/Users/User/PycharmProjects/Gluco_Track/2_Data_Mining/analysis_results",
                0.5
                                )