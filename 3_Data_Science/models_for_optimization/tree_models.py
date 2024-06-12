from sklearn.tree import *

def tree_models(trial, model_name):
    if model_name == "DecisionTreeRegressor":
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
        criterion = trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
        random_state = trial.suggest_categorical('random_state', [None, 42])

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=random_state
        )

    elif model_name == "ExtraTreeRegressor":
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
        criterion = trial.suggest_categorical('criterion', ['poisson', 'friedman_mse', 'absolute_error', 'squared_error'])
        random_state = trial.suggest_categorical('random_state', [None, 42])

        model = ExtraTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=random_state
        )
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model
