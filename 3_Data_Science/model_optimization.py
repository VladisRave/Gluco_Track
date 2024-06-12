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
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
# Функции оптимизации из папки models_for_optimization
from models_for_optimization.ensemble_models import ensemble_models
from models_for_optimization.kernel_ridge_models import kernel_ridge_model
from models_for_optimization.linear_models import linear_models
from models_for_optimization.neighbors_models import knn_regressor
from models_for_optimization.neural_network_models import mlp_regressor
from models_for_optimization.other_models import other_models
from models_for_optimization.svm_models import svm_models
from models_for_optimization.tree_models import tree_models

# Словарь групп моделей
model_groups = {
    'ensemble': [
        AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
        GradientBoostingRegressor, HistGradientBoostingRegressor,
        RandomForestRegressor, LGBMRegressor, XGBRegressor, CatBoostRegressor
    ],
    'linear': [
        BayesianRidge, ElasticNet, ElasticNetCV, HuberRegressor, Lars, LarsCV,
        Lasso, LassoCV, LinearRegression, OrthogonalMatchingPursuit,
        OrthogonalMatchingPursuitCV, PassiveAggressiveRegressor, Ridge,
        RidgeCV, SGDRegressor, TheilSenRegressor, TransformedTargetRegressor
    ],
    'tree': [
        DecisionTreeRegressor, ExtraTreeRegressor
    ],
    'svm': [
        LinearSVR, SVR, NuSVR
    ],
    'neighbors': [
        KNeighborsRegressor
    ],
    'gaussian_process': [
        GaussianProcessRegressor
    ],
    'neural_network': [
        MLPRegressor
    ],
    'dummy': [
        DummyRegressor
    ],
    'kernel_ridge': [
        KernelRidge
    ],
    'other': [
        PoissonRegressor, RANSACRegressor, TweedieRegressor
    ]
}

# Функции для различных групп моделей
def function_usage(trial, model_groups,model_name, X_train, y_train, X_val, y_val):
    # Реализация из сторонней функции
    if model_groups == "ensemble":
        model = ensemble_models(trial,model_name)
    elif model_groups == "linear":
        model = linear_models(trial,model_name)
    elif model_groups == "tree":
        model = tree_models(trial,model_name)
    elif model_groups == "svm":
        model = svm_models(trial,model_name)
    elif model_groups == "neighbors":
        model = knn_regressor(trial,model_name)
    elif model_groups == "gaussian_process":
        model = ensemble_models(trial,model_name)
    elif model_groups == "neural_network":
        model = mlp_regressor(trial,model_name)
    elif model_groups == "dummy":
        model = ensemble_models(trial,model_name)
    elif model_groups == "kernel_ridge":
        model = kernel_ridge_model(trial,model_name)
    elif model_groups == "other":
        model = other_models(trial,model_name)
    else:
        raise ValueError("These model does not exist in this version!")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)

# Функция для выбора и оптимизации моделей
def optimize_models(path, model_names, train_df, val_df,n_trials):
    target_column = "Концентрация_глюкозы"
    X_train = train_df.drop(columns=[target_column], axis=1)
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=[target_column], axis=1)
    y_val = val_df[target_column]
    results = []

    for model_name in model_names:
        for group, models in model_groups.items():
            for model in models:
                if model.__name__ == model_name:
                    study = optuna.create_study(direction='minimize')
                    study.optimize(
                        lambda trial: function_usage(trial, group, model_name, X_train, y_train, X_val, y_val),
                        n_trials=n_trials)

                    # Получение лучших значений оптимизации для модели
                    best_trial = study.best_trial
                    best_params = best_trial.params
                    best_value = best_trial.value

                    # Добавление результатов в список
                    results.append(
                        {'Модель': model_name, 'Лучшее значение метрики': best_value, 'Лучшие параметры': best_params})
                    # Вывод лучших значений оптимизации для модели
                    print(f"Лучшее значение метрики для модели {model_name}:", study.best_value)
                    print(f"Лучшие параметры оптимизации для модели {model_name}:")
                    for param_name, param_value in study.best_params.items():
                        print(f"{param_name}: {param_value}")
                    break

    # Создание DataFrame из результатов и сортировка по убыванию значения метрики
    results_df = pd.DataFrame(results).sort_values(by='Лучшее значение метрики', ascending=True)

    # Сохранение DataFrame в csv файле в той же директории, откуда были взяты данные train и val
    results_csv_path = path + "optimization_results.csv"
    results_df.to_csv(results_csv_path, index=False)

    # Вывод DataFrame с значениями метрик в отсортированном порядке
    print(results_df)

# Ввод данных
path="C:/Users/User/PycharmProjects/Gluco_Track/3_Data_Science/results_computation/"
train_df=pd.read_csv(path+"train_data.csv")
val_df=pd.read_csv(path+"val_data.csv")
# Запуск кода
model_names = pd.read_csv(path+"models_for_optimization.csv")
model_names = list(model_names["Модель"])
optimize_models(path, model_names, train_df, val_df, 10)