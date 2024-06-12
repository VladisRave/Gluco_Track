# Импортирование всех известных моделей
from sklearn.ensemble import *
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def ensemble_models(trial, model_name):
    if model_name == "AdaBoostRegressor":
        # Гиперпараметры для AdaBoostRegressor
        n_estimators = trial.suggest_int('n_estimators', 50, 1000)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 2.0, log=True)
        loss = trial.suggest_categorical('loss', ['linear', 'square', 'exponential'])
        random_state = trial.suggest_int('random_state', 0, 100)

        model = AdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            random_state=random_state
        )

    elif model_name == "BaggingRegressor":
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)
        max_samples = trial.suggest_float('max_samples', 0.1, 1.0)
        max_features = trial.suggest_float('max_features', 0.1, 1.0)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        bootstrap_features = trial.suggest_categorical('bootstrap_features', [True, False])
        oob_score = trial.suggest_categorical('oob_score', [True, False])
        warm_start = trial.suggest_categorical('warm_start', [True, False])
        n_jobs = trial.suggest_int('n_jobs', -1, -1)
        random_state = trial.suggest_int('random_state', 0, 100)

        model = BaggingRegressor(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state
        )

    elif model_name == "CatBoostRegressor":
        iterations = trial.suggest_int('iterations', 100, 1000)
        depth = trial.suggest_int('depth', 1, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5, log=True)
        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1.0, 10.0)
        bagging_temperature = trial.suggest_float('bagging_temperature', 0.0, 1.0)
        random_strength = trial.suggest_float('random_strength', 0.0, 1.0)
        border_count = trial.suggest_int('border_count', 1, 255)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        random_state = trial.suggest_int('random_state', 0, 100)
        verbose=0

        model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=bagging_temperature,
            random_strength=random_strength,
            border_count=border_count,
            subsample=subsample,
            random_state=random_state,
            verbose=verbose
        )

    elif model_name == "ExtraTreesRegressor":
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        criterion = trial.suggest_categorical('criterion', ['poisson', 'friedman_mse', 'absolute_error', 'squared_error'])
        max_depth = trial.suggest_int('max_depth', 1, 50, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5)
        max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 1000, log=True)
        min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.1)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        n_jobs = trial.suggest_int('n_jobs', -1, -1)
        random_state = trial.suggest_int('random_state', 0, 100)
        warm_start = trial.suggest_categorical('warm_start', [True, False])

        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            warm_start=warm_start
        )

    elif model_name == "GradientBoostingRegressor":
        loss = trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber', 'quantile'])
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5, log=True)
        n_estimators = trial.suggest_int('n_estimators', 50, 1000)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        criterion = trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error', 'mse'])
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5)
        max_depth = trial.suggest_int('max_depth', 1, 50, log=True)
        min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.1)
        max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
        alpha = trial.suggest_float('alpha', 0.1, 0.9)
        warm_start = trial.suggest_categorical('warm_start', [True, False])
        random_state = trial.suggest_int('random_state', 0, 100)

        model = GradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            alpha=alpha,
            warm_start=warm_start,
            random_state=random_state
        )

    elif model_name == "HistGradientBoostingRegressor":
        loss = trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'poisson'])
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5, log=True)
        max_iter = trial.suggest_int('max_iter', 50, 1000)
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 1000, log=True)
        max_depth = trial.suggest_int('max_depth', 1, 50, log=True)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        l2_regularization = trial.suggest_float('l2_regularization', 0.0, 1.0)
        max_bins = trial.suggest_int('max_bins', 10, 255)
        warm_start = trial.suggest_categorical('warm_start', [True, False])
        early_stopping = trial.suggest_categorical('early_stopping', [True, False])
        random_state = trial.suggest_int('random_state', 0, 100)

        model = HistGradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_bins=max_bins,
            warm_start=warm_start,
            early_stopping=early_stopping,
            random_state=random_state
        )

    elif model_name == "RandomForestRegressor":
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        criterion = trial.suggest_categorical('criterion', ['mse', 'mae'])
        max_depth = trial.suggest_int('max_depth', 10, 50, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
        min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.1)
        max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None])
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 1000, log=True)
        min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.1)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        oob_score = trial.suggest_categorical('oob_score', [True, False])
        n_jobs = trial.suggest_int('n_jobs', -1, -1)
        random_state = trial.suggest_int('random_state', 0, 100)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state
        )

    elif model_name == "LGBMRegressor":
        num_leaves = trial.suggest_int('num_leaves', 10, 200)
        max_depth = trial.suggest_int('max_depth', -1, 50)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5, log=True)
        n_estimators = trial.suggest_int('n_estimators', 50, 1000)
        min_child_samples = trial.suggest_int('min_child_samples', 5, 100)
        min_child_weight = trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        subsample_freq = trial.suggest_int('subsample_freq', 0, 10)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
        random_state = trial.suggest_int('random_state', 0, 100)

        model = LGBMRegressor(
            num_leaves=num_leaves,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            min_child_weight=min_child_weight,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            verbose = -1
        )

    elif model_name == "XGBRegressor":
        n_estimators = trial.suggest_int('n_estimators', 50, 1000)
        max_depth = trial.suggest_int('max_depth', 1, 50)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5, log=True)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.5, 1.0)
        colsample_bynode = trial.suggest_float('colsample_bynode', 0.5, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
        gamma = trial.suggest_float('gamma', 0.0, 10.0)
        min_child_weight = trial.suggest_float('min_child_weight', 1.0, 10.0)
        random_state = trial.suggest_int('random_state', 0, 100)

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            gamma=gamma,
            min_child_weight=min_child_weight,
            random_state=random_state
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model