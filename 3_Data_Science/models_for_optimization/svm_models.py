from sklearn.svm import *


def svm_models(trial, model_name):
    if model_name == "LinearSVR":
        # Гиперпараметры для LinearSVR
        epsilon = trial.suggest_float('epsilon', 1e-6, 1.0, log=True)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)
        C = trial.suggest_float('C', 1e-6, 1e2, log=True)
        loss = trial.suggest_categorical('loss', ['epsilon_insensitive', 'squared_epsilon_insensitive'])
        fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        intercept_scaling = trial.suggest_float('intercept_scaling', 1e-6, 1e1, log=True)
        dual = trial.suggest_categorical('dual', [True, False])
        max_iter = trial.suggest_int('max_iter', 100, 10000)

        model = LinearSVR(
            epsilon=epsilon,
            tol=tol,
            C=C,
            loss=loss,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            dual=dual,
            max_iter=max_iter
        )

    elif model_name == "SVR":
        # Гиперпараметры для SVR
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        degree = trial.suggest_int('degree', 1, 5)  # используется только для kernel='poly'
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        coef0 = trial.suggest_float('coef0', 0.0, 10.0)  # используется только для kernel='poly' и 'sigmoid'
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)
        C = trial.suggest_float('C', 1e-6, 1e2, log=True)
        epsilon = trial.suggest_float('epsilon', 1e-6, 1.0, log=True)
        shrinking = trial.suggest_categorical('shrinking', [True, False])
        max_iter = trial.suggest_int('max_iter', -1, 10000)

        model = SVR(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            epsilon=epsilon,
            shrinking=shrinking,
            max_iter=max_iter
        )

    elif model_name == "NuSVR":
        # Гиперпараметры для NuSVR
        nu = trial.suggest_float('nu', 0.0, 1.0)
        C = trial.suggest_float('C', 1e-6, 1e2, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        degree = trial.suggest_int('degree', 1, 5)  # используется только для kernel='poly'
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        coef0 = trial.suggest_float('coef0', 0.0, 10.0)  # используется только для kernel='poly' и 'sigmoid'
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)
        shrinking = trial.suggest_categorical('shrinking', [True, False])
        max_iter = trial.suggest_int('max_iter', -1, 10000)

        model = NuSVR(
            nu=nu,
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            shrinking=shrinking,
            max_iter=max_iter
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
