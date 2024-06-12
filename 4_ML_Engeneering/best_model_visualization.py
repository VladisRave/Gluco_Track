# Импорт библиотек
import json
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from  error_grids import plot_zone_accuracy

# Игнорирование ошибок
import warnings
warnings.filterwarnings('ignore')

def optimize_model(path: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Оптимизирует модель на основе данных из файла с гиперпараметрами и сохраняет предсказания.

    Parameters:
    ----------
    path - Путь к файлу с гиперпараметрами.
    train_df - Данные для тренировки модели.
    val_df - Данные для валидации модели.
    test_df - Данные для тестирования модели.

    Возвращает:
    -------
    pd.DataFrame - Датафрейм с предсказанными и теоретическими значениями.
    """
    # Загрузка данных из файла models_for_optimization.csv
    models_df = pd.read_csv(path + "optimization_results.csv")

    # Извлечение первой строки с моделью и гиперпараметрами
    first_model = models_df.iloc[0]
    model_name = first_model["Модель"]
    params = json.loads(first_model["Лучшие параметры"].replace("'", "\""))

    # Вывод параметров и наименования модели
    print("Наименование модели: ", model_name)
    print("Гиперпараметры модели: ", params)

    # Чтение файлов с тренировочными, валидационными и тестовыми данными
    target_column = "Концентрация_глюкозы"
    X_train = train_df.drop(columns=[target_column], axis=1)
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=[target_column], axis=1)
    y_val = val_df[target_column]
    X_test = test_df.drop(columns=[target_column], axis=1)
    y_test = test_df[target_column]

    # Инициализация модели с гиперпараметрами
    model_mapping = {
        'AdaBoostRegressor': AdaBoostRegressor,
        'BaggingRegressor': BaggingRegressor,
        'BayesianRidge': BayesianRidge,
        'CatBoostRegressor': lambda **kwargs: CatBoostRegressor(verbose=0, **kwargs),
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'DummyRegressor': DummyRegressor,
        'ElasticNet': ElasticNet,
        'ElasticNetCV': ElasticNetCV,
        'ExtraTreeRegressor': ExtraTreeRegressor,
        'ExtraTreesRegressor': ExtraTreesRegressor,
        'GaussianProcessRegressor': GaussianProcessRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor,
        'HuberRegressor': HuberRegressor,
        'KernelRidge': KernelRidge,
        'KNeighborsRegressor': KNeighborsRegressor,
        'LGBMRegressor': lambda **kwargs: LGBMRegressor(verbose=-1, **kwargs),
        'Lars': Lars,
        'LarsCV': LarsCV,
        'Lasso': Lasso,
        'LassoCV': LassoCV,
        'LinearRegression': LinearRegression,
        'LinearSVR': LinearSVR,
        'MLPRegressor': MLPRegressor,
        'NuSVR': NuSVR,
        'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit,
        'OrthogonalMatchingPursuitCV': OrthogonalMatchingPursuitCV,
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor,
        'PoissonRegressor': PoissonRegressor,
        'RANSACRegressor': RANSACRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'Ridge': Ridge,
        'RidgeCV': RidgeCV,
        'SVR': SVR,
        'TransformedTargetRegressor': TransformedTargetRegressor,
        'TweedieRegressor': TweedieRegressor,
        'XGBRegressor': XGBRegressor
    }

    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model: {model_name}")

    model = model_mapping[model_name](**params)

    # Обучение модели на данных X_train и y_train
    model.fit(X_train, y_train)
    model.fit(X_val, y_val)

    # Прогнозы на данных X_test
    y_pred = model.predict(X_test)

    # Создание DataFrame с прогнозами и теоретическими значениями
    y_pred_df = pd.DataFrame(data={'Измеренные значения концентрации глюкозы, ммоль/л': y_pred, 'Референсные значения концентрации глюкозы, ммоль/л': y_test}, index=y_test.index)

    # Сохранение DataFrame в текущей директории
    y_pred_df.to_csv('prediction_dataframe.csv')

    return model, X_train

def plot_glucose_concentration(df_standardize: pd.DataFrame, test_df: pd.DataFrame, y_pred_df: pd.DataFrame,
                               start_index: int, end_index: int, step: int = 3006) -> None:
    """
    Функция для построения графика концентрации глюкозы во времени
    на основе основного, валидационного и тестового датасетов.

    Параметры:
    ----------
    test_df - Тестовый датафрейм.
    y_pred_df - Датафрейм с предсказанными значениями для тестового набора.
    start_index - Начальный индекс диапазона.
    end_index - Конечный индекс диапазона.
    step - Шаг времени (по умолчанию 3006).

    Возвращает:
    -------
    None
    """
    # Находим соответствующие индексы в валидационном и тестовом датасетах
    test_indexes = test_df[(test_df.index >= start_index) & (test_df.index <= end_index)].index

    # Строим график
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})

    # Получаем уникальные концентрации глюкозы
    unique_concentrations = df_standardize['Концентрация_глюкозы'].unique()

    # Проходим по каждой уникальной концентрации глюкозы, начиная со второй
    for concentration in unique_concentrations[1:]:
        # Отбираем данные для текущей концентрации из предсказанных значений тестового набора
        test_concentration = y_pred_df[(y_pred_df['Референсные значения концентрации глюкозы, ммоль/л'] == concentration) & y_pred_df.index.isin(test_indexes)]
        # Отбираем данные для текущей концентрации из тестового набора (референсные значения)
        test_reference = test_df[(test_df['Концентрация_глюкозы'] == concentration) & test_df.index.isin(test_indexes)]

        # Строим графики для валидационного и тестового наборов данных
        plt.plot(test_concentration.index*360/step, test_concentration['Измеренные значения концентрации глюкозы, ммоль/л'],
                 color='green', linewidth=3, linestyle='-', label='Предсказанные значения')
        plt.plot(test_reference.index*360/step, test_reference['Концентрация_глюкозы'],
                 color='orange', linewidth=3, linestyle='-', label='Референсные значения')

    # Устанавливаем метки осей
    plt.xlabel('Время измерения, мин')
    plt.ylabel('Концентрация глюкозы, ммоль/л')

    # Устанавливаем ось X с 0
    plt.xlim(start_index*60/step, end_index*60/step)

    # Увеличиваем шрифт наименований на осях
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Добавляем сетку
    plt.grid(True)
    # Показываем график
    plt.show()


def plot_glucose_comparison(result_df: pd.DataFrame) -> None:
    """
    Визуализирует сравнение измеренных и референсных значений концентрации глюкозы с использованием линейной регрессии.

    Параметры:
    ----------
    result_df - DataFrame, содержащий данные для визуализации с колонками
                "Референсные значения концентрации глюкозы, ммоль/л" и
                "Измеренные значения концентрации глюкозы, ммоль/л".

    Возвращает:
    -------
    None
    """
    sns.set(style="whitegrid")

    joint_plot = sns.jointplot(
        data=result_df,
        x="Референсные значения концентрации глюкозы, ммоль/л",
        y="Измеренные значения концентрации глюкозы, ммоль/л",
        kind="scatter",
        color='k',
        s=10
    )

    joint_plot.ax_marg_x.remove()
    joint_plot.ax_marg_y.remove()

    plt.plot(
        result_df["Референсные значения концентрации глюкозы, ммоль/л"],
        result_df["Референсные значения концентрации глюкозы, ммоль/л"],
        color='r',
        linestyle='--',
        linewidth=2,
        label='Прямая пропорциональность'
    )

    sns.regplot(
        x=result_df["Референсные значения концентрации глюкозы, ммоль/л"],
        y=result_df["Измеренные значения концентрации глюкозы, ммоль/л"],
        scatter=False,
        color='g',
        ax=joint_plot.ax_joint,
        line_kws={"linewidth": 1},
        label='Полученная линейная зависимость'
    )

    plt.legend()
    plt.show()


def histogram_glucose(data_df:pd.DataFrame, predicted_column:str, actual_column:str, bins:int=30)->None:
    """
    Построение гистограмм для предсказанных и теоретических значений концентрации глюкозы.

    Параметры:
    ----------
    data_df - DataFrame, содержащий предсказанные и теоретические значения.
    predicted_column - Название столбца с предсказанными значениями.
    actual_column - Название столбца с теоретическими значениями.
    bins - Количество бинов для гистограммы (по умолчанию 30).

    Возвращает:
    -------
    None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Гистограмма для предсказанных значений
    axes[0].hist(data_df[predicted_column], bins=bins, color='blue', alpha=0.7)
    axes[0].set_title('Распределение предсказанных значений')
    axes[0].set_xlabel('Значения концентрации глюкозы, ммоль/л')
    axes[0].set_ylabel('Количество точек, шт')

    # Гистограмма для теоретических значений
    axes[1].hist(data_df[actual_column], bins=bins, color='green', alpha=0.7)
    axes[1].set_title('Распределение теоретических значений')
    axes[1].set_xlabel('Значения концентрации глюкозы, ммоль/л')

    plt.tight_layout()
    plt.show()

def plot_shap_summary(model, X_train: pd.DataFrame) -> None:
    """
    Визуализирует значения SHAP для объяснения модели и оценивает значимость параметров.

    Параметры:
    ----------
    model - Обученная модель регрессии, поддерживающая метод fit и predict.
    X_train - Данные для тренировки, используемые для объяснения модели и визуализации значений SHAP.

    Возвращает:
    -------
    None
    """
    # Создаем объяснение SHAP для модели
    explainer = shap.Explainer(model, X_train)

    # Получаем значения SHAP для всех образцов
    shap_values = explainer(X_train)

    # Визуализируем важность параметров
    plt.title('Оценка значимости параметров в модели')
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    plt.show()

def evaluate_model(data_df:pd.DataFrame)->None:
    """
       Оценка модели на основе предоставленных данных.

       Параметры:
       ----------
       data_df : pd.DataFrame
           Датафрейм, содержащий измеренные и референсные значения концентрации глюкозы.

       Возвращает:
       -------
       None
       """

    # Разделение значений y
    y_true = data_df["Референсные значения концентрации глюкозы, ммоль/л"]
    y_pred = data_df["Измеренные значения концентрации глюкозы, ммоль/л"]

    # Рассчитываем метрики
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Рассчитываем отклонения и статистические параметры
    residuals = y_true - y_pred
    residuals_mean = np.mean(residuals)
    residuals_std = np.std(residuals)
    residuals_min = np.min(residuals)
    residuals_max = np.max(residuals)

    # Создаем DataFrame для красивого вывода метрик и статистических параметров
    metrics_df = pd.DataFrame({
        'Metric': ['R^2 Score', 'Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)',
                   'Root Mean Squared Error (RMSE)',
                   'Residuals Mean', 'Residuals Std', 'Residuals Min', 'Residuals Max'],
        'Value': [r2, mae, mse, rmse, residuals_mean, residuals_std, residuals_min, residuals_max]
    })

    # Выводим DataFrame
    print(metrics_df)

# Указываем путь к файлам
file_path = r'C:\Users\User\PycharmProjects\Gluco_Track\3_Data_Science\model_comparison.py'
path = "C:/Users/User/PycharmProjects/Gluco_Track/3_Data_Science/results_computation/"

# Чтение датафреймов для дальнейшей работы с ними
train_df = pd.read_csv(path + "train_data.csv")
val_df = pd.read_csv(path + "val_data.csv")
test_df = pd.read_csv(path + "test_data.csv")
data_df = pd.read_csv(path + "standartized_df.csv")

# Получение модели и тестовых значений для Х с последующей визуализацией данных
model, X_train = optimize_model(path, train_df, val_df, test_df)

# Построение суммарной визуализации эксперимента
plot_glucose_concentration(data_df, test_df, pd.read_csv("prediction_dataframe.csv"), 0, len(data_df["Концентрация_глюкозы"]))

# Сравнение данных модели машинного обучения и данных референсных
plot_glucose_comparison(pd.read_csv("prediction_dataframe.csv"))

# Распределение гистограммами предиктов и теории
histogram_glucose(pd.read_csv("prediction_dataframe.csv"), "Измеренные значения концентрации глюкозы, ммоль/л", "Референсные значения концентрации глюкозы, ммоль/л")

# Использование полученных предсказанных данных для дальнейшего анализа на сетке Кларка и Паркс
predicted_df = pd.read_csv("prediction_dataframe.csv")
act_arr = predicted_df["Референсные значения концентрации глюкозы, ммоль/л"]
pred_arr = predicted_df["Измеренные значения концентрации глюкозы, ммоль/л"]
evaluate_model(predicted_df)

# Визуализация для двух сеток через гистограммы зон
plot_zone_accuracy(act_arr, pred_arr, mode='clarke', detailed=False)
plot_zone_accuracy(act_arr, pred_arr, mode='parkes', detailed=True, diabetes_type=1)

# Вывод оценок значимости в модели параметров
plot_shap_summary(model, X_train)