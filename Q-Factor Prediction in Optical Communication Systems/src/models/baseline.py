from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)


def get_baseline_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1e-3, max_iter=10_000),
        "ElasticNet": ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=10_000),
        "BayesianRidge": BayesianRidge(),
    }
