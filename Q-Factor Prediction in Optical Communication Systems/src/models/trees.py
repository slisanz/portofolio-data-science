from __future__ import annotations

import numpy as np
import optuna
from tqdm import tqdm
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def default_models(seed: int = 42, verbose: bool = True) -> dict:
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=seed, verbose=1 if verbose else 0
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            verbosity=2 if verbose else 0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1,
            verbose=1 if verbose else -1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=800,
            learning_rate=0.05,
            depth=8,
            random_seed=seed,
            verbose=100 if verbose else False,
        ),
    }


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 50, seed: int = 42):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        )
        model = XGBRegressor(
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=30,
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, preds)))

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    import sys
    pbar = tqdm(total=n_trials, desc="optuna trials", file=sys.stdout)
    study.optimize(objective, n_trials=n_trials, callbacks=[lambda s, t: pbar.update(1)])
    pbar.close()
    return study
