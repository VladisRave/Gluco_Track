from sklearn.neural_network import MLPRegressor

def mlp_regressor(trial, model_name):
    if model_name == "MLPRegressor":
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(100,), (50, 50), (100, 50, 25)])
        activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
        solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])
        alpha = trial.suggest_float('alpha', 1e-5, 1e-2, log=True)
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        tol = trial.suggest_float('tol', 1e-5, 1e-1, log=True)

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol
        )

        return model
