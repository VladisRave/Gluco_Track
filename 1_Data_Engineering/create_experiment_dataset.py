# Импорт библиотек
import os
import csv
import shutil
import math
import time
import pandas as pd
import numpy as np
from scipy.integrate import simpson
from tqdm import tqdm

def add_header_and_rewrite_csv(input_dir: str, output_dir: str) -> None:
    """
    Добавление заголовков в файлы CSV и их перезапись в новую директорию.

    Args:
        input_dir (str): Директория с входными файлами CSV.
        output_dir (str): Директория для сохранения обработанных файлов CSV.
    """
    # Создаём выходную директорию, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Получаем список файлов в указанной директории
    files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # Обрабатываем каждый файл
    for file in files:
        input_file_path = os.path.join(input_dir, file)
        output_file_path = os.path.join(output_dir, file)

        # Открываем существующий файл для чтения
        with open(input_file_path, 'r', newline='') as existing_file:
            existing_content = existing_file.readlines()

        # Проверяем первую строку файла на наличие наименований каждого столбца
        first_line = existing_content[0].strip()
        header = "Порядковый номер точки,Значение АЦП внутреннего фотодиода,Значение АЦП внешнего фотодиода"

        if first_line != header:
            # Открываем файл для записи
            with open(output_file_path, 'w', newline='') as new_file:
                writer = csv.writer(new_file)
                # Добавляем заголовок вверху файла
                writer.writerow([
                    'Порядковый номер точки',
                    'Значение АЦП внутреннего фотодиода',
                    'Значение АЦП внешнего фотодиода'
                ])
                # Записываем остальное содержимое
                for row in existing_content:
                    writer.writerow(row.strip().split(','))
        else:
            # Если файл уже содержит нужный заголовок, копируем его без изменений
            shutil.copy(input_file_path, output_file_path)

def add_num_sample_column(input_dir: str, output_dir: str) -> None:
    """
    Добавление столбца num_sample к файлам CSV и их сохранение в новую директорию.

    Args:
        input_dir (str): Директория с входными файлами CSV.
        output_dir (str): Директория для сохранения обработанных файлов CSV.
    """
    # Создаем выходную директорию, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Список файлов CSV в входной директории
    files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # Обработка каждого файла
    for file in files:
        input_file_path = os.path.join(input_dir, file)
        output_file_path = os.path.join(output_dir, file)

        # Чтение содержимого файла в DataFrame
        df = pd.read_csv(input_file_path, encoding='windows-1251')

        # Добавление столбца num_sample
        temp_dfs = []
        for sample, start_index in enumerate(range(0, len(df), 5001)):
            end_index = min(start_index + 5001, len(df))
            output_signal = df['Значение АЦП внешнего фотодиода'].iloc[start_index:end_index]
            input_signal = df['Значение АЦП внутреннего фотодиода'].iloc[start_index:end_index]

            # Формирование временного DataFrame с добавленным столбцом num_sample
            temp_data = []
            for i, (output_val, input_val) in enumerate(zip(output_signal, input_signal), start=start_index + 1):
                temp_data.append({
                    'Порядковый номер точки': i,
                    'Значение АЦП внешнего фотодиода': output_val,
                    'Значение АЦП внутреннего фотодиода': input_val,
                    'num_sample': sample
                })
            temp_df = pd.DataFrame(temp_data)
            temp_dfs.append(temp_df)

        # Объединение всех временных DataFrame в один
        final_df = pd.concat(temp_dfs, ignore_index=True)

        # Обрезка DataFrame до 300060 строк
        final_df = final_df[:300060]

        # Сохранение DataFrame в новый файл
        final_df.to_csv(output_file_path, index=False)

def calculate_integral(input_dir: str, output_dir: str) -> None:
    """
    Вычисление интегральных сумм для каждого элемента в файлах CSV и их сохранение в новую директорию.

    Args:
        input_dir (str): Директория с входными файлами CSV.
        output_dir (str): Директория для сохранения обработанных файлов CSV.
    """
    # Создаем выходную директорию, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Список файлов CSV в входной директории
    files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # Обработка каждого файла
    for file in files:
        input_file_path = os.path.join(input_dir, file)
        output_file_path = os.path.join(output_dir, file)

        # Чтение содержимого файла в DataFrame
        df = pd.read_csv(input_file_path)

        # Группировка DataFrame по столбцу 'num_sample' и вычисление интегральных сумм
        grouped = df.groupby('num_sample')
        integrals_output = []
        integrals_input = []
        for group_name, group_df in grouped:
            integral_value_output = simpson(y=group_df['Значение АЦП внешнего фотодиода'],
                                            x=group_df['Порядковый номер точки'])
            integral_value_input = simpson(y=group_df['Значение АЦП внутреннего фотодиода'],
                                           x=group_df['Порядковый номер точки'])
            # Расширяем список на длину группы
            integrals_output.extend([integral_value_output] * len(group_df))
            integrals_input.extend([integral_value_input] * len(group_df))

        # Добавление новых столбцов с интегральными суммами в DataFrame
        df['Интегральная сумма внешнего фотодиода'] = integrals_output
        df['Интегральная сумма внутреннего фотодиода'] = integrals_input

        # Сохранение измененного DataFrame в новый файл CSV
        df.to_csv(output_file_path, index=False)

def expand_dataset(csv_dir: str, excel_dir: str, output_dir: str) -> None:
    """
    Объединение результатов эксперимента с описанием эксперимента из файлов CSV и Excel.

    Args:
        csv_dir (str): Директория с файлами CSV.
        excel_dir (str): Директория с файлами Excel.
        output_dir (str): Директория для сохранения объединенных файлов CSV.
    """
    # Получаем список файлов в указанных директориях
    csv_files = [file for file in os.listdir(csv_dir) if file.endswith('.csv')]
    excel_files = [file for file in os.listdir(excel_dir) if file.endswith('.xlsx')]

    # Перебираем каждый файл Excel
    for excel_file in excel_files:
        # Проверяем, есть ли соответствующий файл CSV
        filename = os.path.splitext(excel_file)[0]
        if f"{filename}.csv" in csv_files:
            excel_file_path = os.path.join(excel_dir, excel_file)
            csv_file_path = os.path.join(csv_dir, f"{filename}.csv")
            merged_csv_file_path = os.path.join(output_dir, f"{filename}.csv")

            # Загрузка данных из файлов
            df_excel = pd.read_excel(excel_file_path)
            df_csv = pd.read_csv(csv_file_path)

            # Получение количества записей в исходном файле
            excel_count = len(df_excel)
            csv_count = len(df_csv)

            # Подсчет шага для расширения датасета
            step = math.ceil(csv_count / excel_count)

            # Расширение датасета из Excel
            expanded_rows = []
            for _, row in df_excel.iterrows():
                expanded_rows.extend([row] * step)

            # Создание DataFrame из расширенных строк датасета
            expanded_df_excel = pd.DataFrame(expanded_rows)

            # Добавление столбца 'Порядковый номер точки'
            expanded_df_excel.insert(0, 'Порядковый номер точки', range(1, len(expanded_df_excel) + 1))

            # Объединение по столбцу 'Порядковый номер точки'
            merged_df = pd.merge(expanded_df_excel, df_csv, on='Порядковый номер точки', how='inner')

            # Удаление ненужного столбца
            merged_df.drop(['Минута'], axis=1, inplace=True)

            # Вычисление скользящего среднего для сигналов
            window_size = 5
            merged_df['Значение АЦП внутреннего фотодиода'] = (
                merged_df['Значение АЦП внутреннего фотодиода']
                .rolling(window=window_size, min_periods=1)
                .mean()
            )
            merged_df['Значение АЦП внешнего фотодиода'] = (
                merged_df['Значение АЦП внешнего фотодиода']
                .rolling(window=window_size, min_periods=1)
                .mean()
            )

            # Сохранение объединенного DataFrame в новый файл CSV
            merged_df.to_csv(merged_csv_file_path, index=False)

def extra_pol(input_dir: str, output_dir: str, poly_csv_path: str, temp_csv_path: str, new_column_name: str, step:int=5001)->None:
    """
    Добавление столбца с аппроксимированной температурой лазера или фотодиода.

    Args:
        input_dir (str): Директория с входными файлами CSV.
        output_dir (str): Директория для сохранения файлов с добавленными столбцами CSV.
        poly_csv_path (str): Путь к CSV файлу с данными о степени полинома для каждого num_sample.
        temp_csv_path (str): Путь к CSV файлу с теоретическими температурами для каждого num_sample.
        new_column_name (str): Название нового добавляемого столбца.
        step (int, optional): Шаг для интерполяции значений. По умолчанию 5001.
    """
    # Создаем выходную директорию, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Список файлов CSV в указанной директории
    files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # Обработка каждого файла
    for file in files:
        input_file_path = os.path.join(input_dir, file)
        output_file_path = os.path.join(output_dir, file)

        # Чтение данных из CSV файлов
        df_main = pd.read_csv(input_file_path)
        deg_poly = pd.read_csv(poly_csv_path)
        df_teor_temp = pd.read_csv(temp_csv_path)

        # Обработка каждого уникального num_sample
        for number in df_main['num_sample'].unique():
            # Получение значений температуры из df_teor_temp для текущего num_sample
            y = df_teor_temp[f'minute_{number + 1}'].values

            # Аппроксимация полиномом
            poly_degree = int(deg_poly['poly_minute'][number])  # Степень полинома
            poly_coefficients = np.polyfit(np.linspace(0, 1, len(y)), y, poly_degree)
            poly_function = np.poly1d(poly_coefficients)

            # Получение индексов строк с соответствующим num_sample
            relevant_indices = df_main[df_main['num_sample'] == number].index

            # Вычисление новых значений с использованием интерполяции
            x_new = np.linspace(0, 1, step + 1)
            new_values = poly_function(x_new[:len(relevant_indices)])

            # Корректировка значений на основе охлаждения лазера
            if new_column_name == 'Температура лазера':
                max_temp_laser = df_main['Максимальная температура лазера'][max(relevant_indices)]
            else:
                max_temp_laser = df_main['Максимальная температура фотодиода'][max(relevant_indices)]
            delta = max(new_values) - max_temp_laser
            new_values -= delta

            # Запись новых значений в соответствующие строки
            df_main.loc[relevant_indices, new_column_name] = new_values

        # Сохранение результатов в новый CSV файл
        df_main.to_csv(output_file_path, index=False)

def fourie(input_dir: str, output_dir: str, new_column_name:str)->None:
    """
    Создание столбца Фурье преобразования сигнала.

    Args:
        input_dir (str): Директория с входными файлами CSV.
        output_dir (str): Директория для сохранения файлов с добавленными столбцами CSV.
        new_column_name (str): Название нового столбца для хранения Фурье-преобразования.
    """
    # Создаем выходную директорию, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Список файлов CSV в указанной директории
    files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # Обработка каждого файла
    for file in files:
        input_file_path = os.path.join(input_dir, file)
        output_file_path = os.path.join(output_dir, file)

        # Чтение данных из CSV файла
        df_main = pd.read_csv(input_file_path)

        # Добавление столбца для хранения Фурье-преобразования
        df_main[new_column_name + ' внутреннего фотодиода'] = np.nan
        df_main[new_column_name + ' внешнего фотодиода'] = np.nan

        # Вычисление Фурье-преобразования для каждого num_sample
        for number in df_main['num_sample'].unique():
            relevant_indices = df_main[df_main['num_sample'] == number].index

            # Получение данных для вычисления Фурье-преобразования
            fourie_inside = df_main.loc[relevant_indices, 'Значение АЦП внутреннего фотодиода'].values
            fourie_outside = df_main.loc[relevant_indices, 'Значение АЦП внешнего фотодиода'].values

            # Преобразование Фурье для внутреннего фотодиода
            fft_inside = np.fft.fft(fourie_inside)
            module_fft_inside = np.abs(fft_inside * np.conj(fft_inside))

            # Преобразование Фурье для внешнего фотодиода
            fft_outside = np.fft.fft(fourie_outside)
            module_fft_outside = np.abs(fft_outside * np.conj(fft_outside))

            # Запись значений в соответствующие строки
            df_main.loc[relevant_indices, new_column_name + ' внутреннего фотодиода'] = module_fft_inside
            df_main.loc[relevant_indices, new_column_name + ' внешнего фотодиода'] = module_fft_outside

        # Сохранение результатов в новый CSV файл
        df_main.to_csv(output_file_path, index=False)

def time_exp_add(input_dir: str, output_dir: str, step:int=10/5001)->None:
    """
    Добавление времени с момента эксперимента.

    Args:
        input_dir (str): Директория с входными файлами CSV.
        output_dir (str): Директория для сохранения файлов с добавленными столбцами CSV.
        step (float, optional): Шаг времени. По умолчанию 10 / 5001.
    """
    # Создаем выходную директорию, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Список файлов CSV в указанной директории
    files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # Обработка каждого файла
    new_column_name = 'Время с начала эксперимента'
    for file in files:
        input_file_path = os.path.join(input_dir, file)
        output_file_path = os.path.join(output_dir, file)

        # Чтение данных из CSV файла
        df_main = pd.read_csv(input_file_path)

        # Добавление столбца для хранения времени
        df_main['Время с начала эксперимента'] = np.nan

        # Вычисление времени с момента начала эксперимента для каждого num_sample
        for number in df_main['num_sample'].unique():
            relevant_indices = df_main[df_main['num_sample'] == number].index

            # Получение данных для вычисления времени с момента эксперимента
            for index in relevant_indices:
                df_main.loc[index, new_column_name] = step * (index + 50 * (number))

        # Сохранение результатов в новый CSV файл
        df_main.to_csv(output_file_path, index=False)

def remove_column_and_save(input_directory: str, output_directory: str)->None:
    """
    Удаление лишних столбцов из файлов CSV и сохранение результатов.

    Args:
        input_directory (str): Директория с входными файлами CSV.
        output_directory (str): Директория для сохранения файлов без лишних столбцов CSV.
    """
    # Создаем выходную директорию, если она не существует
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Список файлов CSV в указанной директории
    files = [file for file in os.listdir(input_directory) if file.endswith('.csv')]

    # Обработка каждого файла
    for file in files:
        input_file_path = os.path.join(input_directory, file)
        output_file_path = os.path.join(output_directory, file)

        # Загружаем данные из CSV файла в DataFrame
        df = pd.read_csv(input_file_path)

        # Удаляем лишние столбцы, если они присутствуют
        columns_to_remove = ['num_sample', 'Порядковый номер точки', 'Максимальная температура фотодиода', 'Максимальная температура лазера']
        for column in columns_to_remove:
            if column in df.columns:
                df = df.drop(columns=[column])

        # Перемещение столбца с целевой переменной 'Концентрация глюкозы' в конец
        if 'Концентрация глюкозы' not in df.columns:
            print("Столбец с целевой переменной отсутствует в DataFrame")
            return df

        # Создаем список из названий столбцов DataFrame
        columns = list(df.columns)

        # Перемещаем столбец с целевой переменной в конец списка
        columns.remove('Концентрация глюкозы')
        columns.append('Концентрация глюкозы')

        # Создаем новый DataFrame с измененным порядком столбцов
        df = df[columns]

        # Сохраняем новый датафрейм без лишних столбцов в новый CSV файл
        df.to_csv(output_file_path, index=False)

# Начало отсчёта времени работы кода
start_time = time.time()

print("Осуществляется процесс обработки данных...")
# Создаем tqdm прогресс-бар
with tqdm(total=9) as pbar:
    # 1) Добавление названия столбцов в датафрейм
    add_header_and_rewrite_csv(
                        "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/experimental_data",
                        "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes"
                                )
    pbar.update(1)

    # 2) Добавление столбца num_sample к файлам
    input_dir = "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes"
    output_dir = "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes"
    add_num_sample_column(
                        "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes",
                        "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes"
                        )
    pbar.update(1)

    # 3) Вычисление интегральных сумм для входного выходного АЦП
    calculate_integral(
                    "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes",
                    "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes"
                        )
    pbar.update(1)

    # 4) Объединение результатов эксперимента с описанием эксперимента
    csv_dir = "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes"
    xlsx_dir = "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/description_of_the_experiment"
    result_dir = "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes"
    expand_dataset(
                "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes",
                "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/description_of_the_experiment",
                "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes"
                    )
    pbar.update(1)

    # 5) Экстраполяция данных для лазера
    extra_pol(
        "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes",
        "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes",
        "C:/Users/User/PycharmProjects/Gluco_Track/1_Data_Engineering/temperature/Степени_полиномов_лазера.csv",
        "C:/Users/User/PycharmProjects/Gluco_Track/1_Data_Engineering/temperature/Аппроксимация_эксперимента_лазера.csv",
        "Температура лазера"
    )
    pbar.update(1)

    # 6) Экстраполяция данных для фотодиода
    extra_pol(
        "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes",
        "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes",
        "C:/Users/User/PycharmProjects/Gluco_Track/1_Data_Engineering/temperature/Степени_полиномов_фотодиода.csv",
        "C:/Users/User/PycharmProjects/Gluco_Track/1_Data_Engineering/temperature/Аппроксимация_эксперимента_фотодиода.csv",
        "Температура фотодиода"
    )
    pbar.update(1)

    # 7) Добавление Фурье-образов АЦП внутреннего и внешнего фотодиода
    fourie(
        "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes",
        "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes",
        "Фурье образ"
    )
    pbar.update(1)

    # 8) Добавление времени с начала эксперимента с учётом остывания лазера
    time_exp_add(
        "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes",
        "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes"
    )
    pbar.update(1)

    # 9) Удаление вспомогательных столбцов датафрейма
    remove_column_and_save("C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes",
                           "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframes"
                           )
    pbar.update(1)

print("Время выполнения кода:", round(time.time() - start_time, 2), "сек")
