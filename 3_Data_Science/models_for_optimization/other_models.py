from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import PoissonRegressor, RANSACRegressor, TweedieRegressor

def other_models(trial, model_name):
    if model_name == "PoissonRegressor":
        # Гиперпараметры для PoissonRegressor
        alpha = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
        fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        max_iter = trial.suggest_int('max_iter', 100, 10000)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)

        model = PoissonRegressor(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol
        )

    elif model_name == "RANSACRegressor":
        # Гиперпараметры для RANSACRegressor
        min_samples = trial.suggest_int('min_samples', 1, 100)
        residual_threshold = trial.suggest_float('residual_threshold', 0.01, 10.0)
        max_trials = trial.suggest_int('max_trials', 10, 1000)
        loss = trial.suggest_categorical('loss', ['absolute_loss', 'squared_loss'])

        model = RANSACRegressor(
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            loss=loss
        )

    elif model_name == "TweedieRegressor":
        # Гиперпараметры для TweedieRegressor
        power = trial.suggest_float('power', 0.0, 2.0)
        alpha = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
        fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        max_iter = trial.suggest_int('max_iter', 100, 10000)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)

        model = TweedieRegressor(
            power=power,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
