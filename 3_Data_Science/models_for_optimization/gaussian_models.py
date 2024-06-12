from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct

def gaussian_models(trial, model_name):
    if model_name == "GaussianProcessRegressor":
        kernel = trial.suggest_categorical('kernel', ['rbf', 'matern', 'rational_quadratic', 'exp_sine_squared',
                                                      'dot_product'])  # выбор ядра
        alpha = trial.suggest_float('alpha', 1e-10, 1e-1, log=True)  # параметр шума
        length_scale = trial.suggest_float('length_scale', 1e-1, 1e1,
                                           log=True)  # параметр для RBF, matern и rational quadratic ядер
        nu = trial.suggest_float('nu', 0.5, 5.0)  # параметр для matern и rational quadratic ядер

        if kernel == 'rbf':
            kernel_obj = RBF(length_scale=length_scale)
        elif kernel == 'matern':
            kernel_obj = Matern(length_scale=length_scale, nu=nu)
        elif kernel == 'rational_quadratic':
            kernel_obj = RationalQuadratic(length_scale=length_scale, alpha=alpha)
        elif kernel == 'exp_sine_squared':
            kernel_obj = ExpSineSquared(length_scale=length_scale, periodicity=1.0)
        elif kernel == 'dot_product':
            kernel_obj = DotProduct(sigma_0=1.0)
        else:
            raise ValueError("Unknown kernel type!")

        model = GaussianProcessRegressor(kernel=kernel_obj, alpha=alpha)

        return model