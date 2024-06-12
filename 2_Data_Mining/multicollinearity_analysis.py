import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

def analyze_files(input_dir: str, output_dir: str) -> None:
    """
    Анализ файлов в указанной директории и сохранение отчёта.

    Параметры:
    ----------
    input_dir (str): Путь к входной директории, содержащей файлы для анализа.
    output_dir (str): Путь к выходной директории для сохранения отчета.

    Возвращает:
    ----------
    None
    """
    # Получаем список файлов в указанной директории
    files = [file for file in os.listdir(input_dir) if file.endswith(".csv")]

    # Создаем пустой DataFrame с нужными столбцами
    dataframe_removed_columns = pd.DataFrame(columns=["Имя файла", "Список удалённых столбцов"])

    # Обрабатываем каждый файл
    for file in tqdm(files, desc="Processing files"):
        input_file_path = os.path.join(input_dir, file)
        # Извлекаем имя файла с расширением и убираем расширение .csv
        file_name = os.path.splitext(os.path.basename(input_file_path))[0]

        # Открываем существующий файл для чтения
        df = pd.read_csv(input_file_path)
        # Переименовываем столбцы, чтобы они были валидными идентификаторами
        df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")

        # Выделяем X и стандартизуем его
        target_variable = "Концентрация_глюкозы"
        labels = df.drop(columns=[target_variable]).columns.tolist()
        df_standardize = standardize_data(df, target_variable)
        removed_columns_list = remove_collinear_columns(df_standardize, target_variable, *labels)

        # Добавляем данные о файле и удаленных столбцах в DataFrame
        row = pd.DataFrame({"Имя файла": [file_name], "Список удалённых столбцов": [removed_columns_list]})
        dataframe_removed_columns = pd.concat([dataframe_removed_columns, row], ignore_index=True)

    # Сохраняем результирующий DataFrame в файл в выходной директории
    output_file_path = os.path.join(output_dir, "removed_columns_report.csv")
    dataframe_removed_columns.to_csv(output_file_path, index=False)


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

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_standardized = (X - mean) / std
    X_standardized = X_standardized / np.sqrt(np.sum(X_standardized ** 2, axis=0))

    scaled_df = deepcopy(X_standardized)
    scaled_df[target_variable] = y.values

    return scaled_df

def remove_collinear_columns(df: pd.DataFrame, target_variable: str, *initial_labels) -> list:
    """
    Удаление коллинеарных столбцов из DataFrame.

    Параметры:
    ----------
    df (pd.DataFrame): Входной DataFrame для анализа.
    target_variable (str): Название целевой переменной в DataFrame.
    initial_labels (list): Список начальных меток столбцов.

    Возвращает:
    ----------
    list-Список удаленных коллинеарных столбцов.
    """
    removed_columns = []
    labels = list(initial_labels)
    while True:
        vif_result = vif(df, target_variable, *labels)
        if any(vif_result["VIF"] > 5):
            condind_matrix = vdp_matrix(labels, df[labels])
            if condind_matrix["condInd"].max() > 10:
                row = condind_matrix["condInd"].idxmax()
                s = condind_matrix.loc[row, labels]
                collinear_columns = s[s > 0.5].index.values
                collinear_columns_in_vif = [col for col in collinear_columns if col in vif_result["variable"].values]

                if len(collinear_columns_in_vif) > 1:
                    column_to_drop = max(collinear_columns_in_vif,
                                         key=lambda x: vif_result.loc[vif_result["variable"] == x, "VIF"].values[0])
                elif len(collinear_columns_in_vif) == 1:
                    column_to_drop = collinear_columns_in_vif[0]
                else:
                    print("No collinear columns found in VIF. Exiting loop.")
                    break

                removed_columns.append(column_to_drop)
                df.drop(columns=[column_to_drop], inplace=True)
                labels = [label for label in labels if label != column_to_drop]
            else:
                break
        else:
            break

    return removed_columns


def vif(df: pd.DataFrame, target_variable: str, *predictors) -> pd.DataFrame:
    """
    Вычисление фактора инфляции дисперсии (VIF) для переменных в DataFrame.

    Параметры:
    ----------
    df (pd.DataFrame): Входной DataFrame для анализа.
    target_variable (str): Название целевой переменной в DataFrame.
    predictors (list): Список предикторов для вычисления VIF.

    Возвращает:
    ----------
    pd.DataFrame-DataFrame с результатами VIF.
    """
    formula = f"{target_variable} ~ {" + ".join(predictors)}"
    y, X = dmatrices(formula, data=df, return_type="dataframe")
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["variable"] = X.columns
    vif = vif.iloc[1:].reset_index(drop=True)

    return vif

def vdp_matrix(labels: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисление матрицы вектора диагональных пар (VDP) для переменных в DataFrame.

    Параметры:
    ----------
    labels (list): Список меток столбцов для анализа.
    df (pd.DataFrame): Входной DataFrame для анализа.

    Возвращает:
    ----------
    pd.DataFrame-DataFrame с результатами VDP.
    """
    X = df[labels]
    U, S, V = np.linalg.svd(X, full_matrices=False)
    lambda_ = S
    condind = S[0] / lambda_

    phi_mat = (V.T * V.T) / (lambda_ ** 2)
    phi = np.sum(phi_mat, axis=1).reshape(-1, 1)
    vdp = np.divide(phi_mat, phi).T

    vdp_df = pd.DataFrame(data=vdp, columns=labels)
    vdp_df.insert(0, 'sValue', S)
    vdp_df.insert(1, 'condInd', condind)
    vdp_df = vdp_df.set_index('sValue')
    pd.options.display.float_format = '{:.5f}'.format

    return vdp_df

# Пример использования функции
analyze_files(
        "C:/Users/User/PycharmProjects/Gluco_Track/0_Data/aggregated_dataframe_interpolated",
        "C:/Users/User/PycharmProjects/Gluco_Track/2_Data_Mining/analysis_results"
                )
