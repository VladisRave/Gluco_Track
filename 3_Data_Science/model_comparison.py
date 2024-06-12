# Импорт всех необходимых библиотек для работы
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Импорт бара загрузки
from tqdm import tqdm
# Импортирование метрик
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Импортирование всех известных моделей
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from copy import deepcopy
# Игнорирование ошибок
import warnings
warnings.filterwarnings('ignore')

def standardize_data(df: pd.DataFrame, output_file_path: str, target_variable: str='Концентрация_глюкозы') -> pd.DataFrame:
    """
    Стандартизация данных.

    Параметры:
    ----------
    df (pd.DataFrame): Входной DataFrame для стандартизации.
    target_variable (str): Название целевой переменной в DataFrame.
    output_file_path (str): Название выходной дирректории где будет сохранён стандартизованный DataFrame

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

    scaled_df.to_csv(output_file_path, index=False)

    return scaled_df


def plot_sum_graph(df: pd.DataFrame, target_value: str, step: int = 3006) -> None:
    """
    Функция для построения графика суммарной концентрации глюкозы во времени.

    Параметры:
    df (DataFrame): Датафрейм с данными
    target_value (str): Название столбца с целевым значением
    step (int, optional): Шаг времени (по умолчанию 3006)

    Возвращает:
    ----------
    None
    """
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})

    indx = df.index / step
    plt.plot(indx, df[target_value], color='blue')

    plt.xlabel('Время измерения, ч')
    plt.ylabel('Концентрация глюкозы, ммоль/л')
    plt.xlim(0, indx[-1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.show()


def split_dataframes_by_glucose_concentration(df: pd.DataFrame, output_dir: str,target_column: str = 'Концентрация_глюкозы'):
    """
    Функция для разделения датафрейма на тренировочный, валидационный и тестовый
    датасеты по уникальным значениям концентрации глюкозы.

    Параметры:
    df (DataFrame): Датафрейм с данными
    target_column (str, optional): Название столбца с целевым значением (по умолчанию 'Концентрация глюкозы')
    output_dir (str): Название выходной дирректории где будет сохранены все файлы

    Возвращает:
    tuple: Три датафрейма (train_df, val_df, test_df)
    """
    unique_concentrations = df[target_column].unique()

    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for concentration in unique_concentrations:
        df_concentration = df[df[target_column] == concentration]

        train_size = int(0.7 * df_concentration.shape[0])
        val_size = int(0.1 * df_concentration.shape[0])

        train_data = df_concentration.iloc[:train_size]
        val_data = df_concentration.iloc[train_size:train_size + val_size]
        test_data = df_concentration.iloc[train_size + val_size:]

        train_df = pd.concat([train_df, train_data])
        val_df = pd.concat([val_df, val_data])
        test_df = pd.concat([test_df, test_data])

    train_df.to_csv(output_dir + "train_data.csv", index=False)
    val_df.to_csv(output_dir + "val_data.csv", index=False)
    test_df.to_csv(output_dir + "test_data.csv", index=False)

    return train_df, val_df, test_df


def plot_glucose_concentration(df_standardize: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                               start_index: int, end_index: int, step: int = 3006):
    """
    Функция для построения графика концентрации глюкозы во времени
    на основе основного, валидационного и тестового датасетов.

    Параметры:
    df_standardize (DataFrame): Основной датафрейм с данными
    val_df (DataFrame): Валидационный датафрейм
    test_df (DataFrame): Тестовый датафрейм
    start_index (int): Начальный индекс диапазона
    end_index (int): Конечный индекс диапазона
    step (int, optional): Шаг времени (по умолчанию 3006)
    """
    val_indexes = val_df[(val_df.index >= start_index) & (val_df.index <= end_index)].index
    test_indexes = test_df[(test_df.index >= start_index) & (test_df.index <= end_index)].index

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})

    plt.plot((df_standardize.index[start_index:end_index]) * 60 / step,
             df_standardize['Концентрация_глюкозы'][start_index:end_index],
             color='blue', linewidth=3, linestyle='--', label='Основной набор')

    unique_concentrations = df_standardize['Концентрация_глюкозы'].unique()

    for concentration in unique_concentrations[1:]:
        val_concentration = val_df[(val_df['Концентрация_глюкозы'] == concentration) & val_df.index.isin(val_indexes)]
        test_concentration = test_df[
            (test_df['Концентрация_глюкозы'] == concentration) & test_df.index.isin(test_indexes)]

        plt.plot(test_concentration.index * 60 / step, test_concentration['Концентрация_глюкозы'],
                 color='green', linewidth=4, linestyle='-', label='Тестовый набор')
        plt.plot(val_concentration.index * 60 / step, val_concentration['Концентрация_глюкозы'],
                 color='red', marker='o', linestyle='', label='Валидационный набор')

    plt.xlabel('Время измерения, мин')
    plt.ylabel('Концентрация глюкозы, ммоль/л')
    plt.xlim(start_index * 60 / step, end_index * 60 / step)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.show()

def evaluate_models(train_df, test_df, target_column='Концентрация_глюкозы', output_dir='results'):
    """
    Функция для оценки моделей регрессии на данных.

    Параметры:
    train_df (DataFrame): Тренировочный датафрейм
    test_df (DataFrame): Тестовый датафрейм
    target_column (str): Название целевого столбца (по умолчанию 'Концентрация глюкозы')
    output_dir (str): Директория для сохранения результатов (по умолчанию 'results')

    Возвращает:
    None
    """
    X_train = train_df.drop(columns=[target_column], axis=1)
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column], axis=1)
    y_test = test_df[target_column]

    models = [
        AdaBoostRegressor(),
        BaggingRegressor(),
        BayesianRidge(),
        CatBoostRegressor(verbose=False),
        DecisionTreeRegressor(),
        DummyRegressor(),
        ElasticNet(),
        ElasticNetCV(),
        ExtraTreeRegressor(),
        ExtraTreesRegressor(),
        GaussianProcessRegressor(),
        GradientBoostingRegressor(),
        HistGradientBoostingRegressor(),
        HuberRegressor(),
        KernelRidge(),
        KNeighborsRegressor(),
        LGBMRegressor(verbose=-1),
        Lars(),
        LarsCV(),
        Lasso(),
        LassoCV(),
        LinearRegression(),
        LinearSVR(),
        MLPRegressor(),
        NuSVR(),
        OrthogonalMatchingPursuit(),
        OrthogonalMatchingPursuitCV(),
        PassiveAggressiveRegressor(),
        PoissonRegressor(),
        RANSACRegressor(),
        RandomForestRegressor(),
        Ridge(),
        RidgeCV(),
        SVR(),
        TransformedTargetRegressor(),
        TweedieRegressor(),
        XGBRegressor(),
    ]

    results = pd.DataFrame(columns=['Модель', 'R2', 'MAE', 'RMSE', 'Время, с'])

    for model in tqdm(models, desc="Оценка моделей"):
        start_time = time.time()
        model_name = model.__class__.__name__
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_test = np.array(y_test)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        new_row = pd.DataFrame(
            {'Модель': [model_name], 'R2': [r2], 'MAE': [mae], 'RMSE': [rmse], 'Время, с': [elapsed_time]})
        results = pd.concat([results, new_row], ignore_index=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sorted_df = results.sort_values(by=['MAE', 'RMSE', 'Время, с'], axis=0, ascending=True)
    sorted_df = sorted_df.sort_values(by=['R2'], axis=0, ascending=False)
    sorted_df = sorted_df.reset_index(drop=True)
    sorted_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)


def main(input_dir: str, output_dir: str, target_variable: str = "Концентрация_глюкозы"):
    """
    Главная функция для запуска всех скриптов по порядку.

    Параметры:
    input_dir (str): Директория с входными CSV файлами.
    output_dir (str): Директория для сохранения результатов.
    target_variable (str, optional): Название целевой переменной в DataFrame (по умолчанию 'Концентрация_глюкозы').

    Возвращает:
    None
    """
    df_list = []
    files = [file for file in os.listdir(input_dir) if file.endswith(".csv")]

    for file in files:
        input_file_path = os.path.join(input_dir, file)
        # Чтение файла
        df = pd.read_csv(input_file_path)
        df.columns = df.columns.str.replace(" ", "_").str.replace("-", "_")

        # Добавление в массив данных
        df_list.append(df)

    # Объединение всех датафреймов в один
    merged_df = pd.concat(df_list, ignore_index=True)

    # Стандартизация данных в датафрейме кроме концентрации глюкозы
    merged_df = standardize_data(merged_df,
                                output_dir + "/standartized_df.csv",
                                target_variable)

    # Построение суммарного временного ряда
    plot_sum_graph(merged_df, target_variable)
    # Разделение на три типа данных для дальнейшего применения
    train_df, val_df, test_df = split_dataframes_by_glucose_concentration(merged_df,"C:/Users/User/PycharmProjects/Gluco_Track/3_Data_Science/results_computation/")
    # Построение куска даты с разделением на стартовый и конечный индексы
    plot_glucose_concentration(merged_df, val_df, test_df, start_index=4400, end_index=6300)
    # Использование моделей для определения наиболее пригодных к поставленной задаче
    evaluate_models(train_df, test_df, target_column=target_variable, output_dir=output_dir)

# Пример вызова главной функции
main("C:/Users/User/PycharmProjects/Gluco_Track/3_Data_Science/clean_dataframes",
     "C:/Users/User/PycharmProjects/Gluco_Track/3_Data_Science/results_computation"
     )
