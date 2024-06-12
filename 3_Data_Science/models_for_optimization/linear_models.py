from sklearn.linear_model import *
from sklearn.compose import TransformedTargetRegressor

def linear_models(trial, model_name):
    if model_name == "BayesianRidge":
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)
        alpha_1 = trial.suggest_float('alpha_1', 1e-6, 1e-1, log=True)
        alpha_2 = trial.suggest_float('alpha_2', 1e-6, 1e-1, log=True)
        lambda_1 = trial.suggest_float('lambda_1', 1e-6, 1e-1, log=True)
        lambda_2 = trial.suggest_float('lambda_2', 1e-6, 1e-1, log=True)
        alpha_init = trial.suggest_float('alpha_init', 1e-6, 1e-1, log=True)
        lambda_init = trial.suggest_float('lambda_init', 1e-6, 1e-1, log=True)

        model = BayesianRidge(
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            alpha_init=alpha_init,
            lambda_init=lambda_init
        )

    elif model_name == "ElasticNet":
        alpha = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol
        )

    elif model_name == "ElasticNetCV":
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        n_alphas = trial.suggest_int('n_alphas', 50, 200)
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)

        model = ElasticNetCV(
            l1_ratio=l1_ratio,
            n_alphas=n_alphas,
            max_iter=max_iter,
            tol=tol
        )

    elif model_name == "HuberRegressor":
        epsilon = trial.suggest_float('epsilon', 1.1, 2.0)
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        alpha = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)

        model = HuberRegressor(
            epsilon=epsilon,
            max_iter=max_iter,
            alpha=alpha,
            tol=tol
        )

    elif model_name == "Lars":
        n_nonzero_coefs = trial.suggest_int('n_nonzero_coefs', 1, 500)
        eps = trial.suggest_float('eps', 1e-6, 1e-1, log=True)

        model = Lars(
            n_nonzero_coefs=n_nonzero_coefs,
            eps=eps
        )

    elif model_name == "LarsCV":
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        eps = trial.suggest_float('eps', 1e-6, 1e-1, log=True)

        model = LarsCV(
            max_iter=max_iter,
            eps=eps
        )

    elif model_name == "Lasso":
        alpha = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)

        model = Lasso(
            alpha=alpha,
            max_iter=max_iter,
            tol=tol
        )

    elif model_name == "LassoCV":
        n_alphas = trial.suggest_int('n_alphas', 50, 200)
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)

        model = LassoCV(
            n_alphas=n_alphas,
            max_iter=max_iter,
            tol=tol
        )

    elif model_name == "LinearRegression":
        fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        n_jobs = trial.suggest_int('n_jobs', -1, -1)

        model = LinearRegression(
            fit_intercept=fit_intercept,
            n_jobs=n_jobs
        )

    elif model_name == "OrthogonalMatchingPursuit":
        n_nonzero_coefs = trial.suggest_int('n_nonzero_coefs', 1, 500)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)

        model = OrthogonalMatchingPursuit(
            n_nonzero_coefs=n_nonzero_coefs,
            tol=tol
        )


    elif model_name == "OrthogonalMatchingPursuitCV":
        cv = trial.suggest_categorical('cv', [None, 3, 5, 10])
        verbose = trial.suggest_categorical('verbose', [True, False])
        n_jobs = trial.suggest_categorical('n_jobs', [None, 1, 2, 4])

        model = OrthogonalMatchingPursuitCV(
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs
        )


    elif model_name == "PassiveAggressiveRegressor":
        C = trial.suggest_float('C', 1e-6, 1e2, log=True)
        fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)
        early_stopping = trial.suggest_categorical('early_stopping', [True, False])
        validation_fraction = trial.suggest_float('validation_fraction', 0.01, 0.5)
        n_iter_no_change = trial.suggest_int('n_iter_no_change', 1, 20)

        model = PassiveAggressiveRegressor(
            C=C,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change
        )

    elif model_name == "Ridge":
        alpha = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
        fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)
        solver = trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])

        model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            tol=tol,
            solver=solver
        )

    elif model_name == "RidgeCV":
        alphas = trial.suggest_categorical('alphas', [[0.1, 1.0, 10.0], [0.01, 0.1, 1.0, 10.0, 100.0]])
        fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        scoring = trial.suggest_categorical('scoring', [None, 'neg_mean_squared_error', 'r2'])

        model = RidgeCV(
            alphas=alphas,
            fit_intercept=fit_intercept,
            scoring=scoring
        )

    elif model_name == "SGDRegressor":
        loss = trial.suggest_categorical('loss', ['squared_error', 'huber', 'epsilon_insensitive',
                                                  'squared_epsilon_insensitive'])
        penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])
        alpha = trial.suggest_float('alpha', 1e-6, 1e2, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)
        epsilon = trial.suggest_float('epsilon', 1e-6, 1.0)
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'])
        eta0 = trial.suggest_float('eta0', 1e-6, 1e-1, log=True)
        power_t = trial.suggest_float('power_t', 0.0, 1.0)
        early_stopping = trial.suggest_categorical('early_stopping', [True, False])
        validation_fraction = trial.suggest_float('validation_fraction', 0.01, 0.5)
        n_iter_no_change = trial.suggest_int('n_iter_no_change', 1, 20)

        model = SGDRegressor(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            epsilon=epsilon,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change
        )

    elif model_name == "TheilSenRegressor":
        fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
        max_subpopulation = trial.suggest_int('max_subpopulation', 1e3, 1e6)
        n_subsamples = trial.suggest_int('n_subsamples', 1, 1e6)
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        tol = trial.suggest_float('tol', 1e-6, 1e-1, log=True)

        model = TheilSenRegressor(
            fit_intercept=fit_intercept,
            max_subpopulation=max_subpopulation,
            n_subsamples=n_subsamples,
            max_iter=max_iter,
            tol=tol
        )

    elif model_name == "TransformedTargetRegressor":
        func = trial.suggest_categorical('func', [None, 'np.log', 'np.exp'])
        inverse_func = trial.suggest_categorical('inverse_func', [None, 'np.log', 'np.exp'])

        model = TransformedTargetRegressor(
            func=func,
            inverse_func=inverse_func
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
