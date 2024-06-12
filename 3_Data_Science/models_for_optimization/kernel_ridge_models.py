from sklearn.kernel_ridge import KernelRidge

def kernel_ridge_model(trial, model_name):
    if model_name == "KernelRidge":
        alpha = trial.suggest_float('alpha', 1e-5, 1e2, log=True)  # параметр регуляризации
        kernel = trial.suggest_categorical('kernel', ['linear', 'polynomial', 'rbf', 'laplacian', 'sigmoid'])  # выбор ядра
        gamma = trial.suggest_float('gamma', 1e-5, 1e2, log=True)  # параметр для RBF, laplacian и sigmoid ядер
        degree = trial.suggest_int('degree', 2, 5)  # степень полинома, используется только для polynomial ядра
        coef0 = trial.suggest_float('coef0', 0.0, 10.0)  # свободный член, используется только для polynomial и sigmoid ядер

        model = KernelRidge(
            alpha=alpha,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0
        )

        return model