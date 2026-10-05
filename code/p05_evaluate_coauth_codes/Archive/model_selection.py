import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
import logging

# Set up logging to model_selection.md
logging.basicConfig(
    filename='../reports/model_selection_grid.md',
    level=logging.INFO,
    format='%(message)s',
    filemode='w'  # Overwrite file each run; use 'a' to append
)
logging.info("# Model Selection Results\n")


# Custom feature selector based on correlation
class TopKCorrelationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=100):
        self.k = k
        self.selected_features = None

    def fit(self, X, y):
        corrs = np.array([abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])])
        self.selected_features = np.argsort(corrs)[-self.k:]
        return self

    def transform(self, X):
        return X[:, self.selected_features]


# Main class for Embedded Feature Selection
class EFS:
    def __init__(self, npz_file='../data/processed_data/all.npz'):
        self.npz_file = npz_file
        self.fig_dir = 'figures'
        os.makedirs(self.fig_dir, exist_ok=True)
        self.split_file = '../data/processed_data/traintest/holdout_split.npz'
        self.outer_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
        self.scoring = "r2"  # Use R² as the primary scoring metric
        self.save_holdout()
        self.load_split()

    def save_holdout(self, holdout_size=0.2):
        """Save training and holdout splits to a .npz file."""
        loader = np.load(self.npz_file)
        holdout_data = {}
        for table_name in ['FTER', 'FTMR', 'DTER', 'DTMR', 'TTER', 'TTMR']:
            data_array = loader[table_name]
            X, Y = data_array[:, 1:], data_array[:, 0]
            indices = np.arange(len(Y))
            np.random.seed(42)
            np.random.shuffle(indices)
            n_holdout = int(len(Y) * holdout_size)
            holdout_idx = indices[:n_holdout]
            train_idx = indices[n_holdout:]
            holdout_data[table_name] = {
                'train': {'X': X[train_idx], 'Y': Y[train_idx]},
                'holdout': {'X': X[holdout_idx], 'Y': Y[holdout_idx]}
            }
        np.savez_compressed(
            self.split_file,
            **{f"{k}_{v}_X": holdout_data[k][v]['X'] for k in holdout_data for v in ['train', 'holdout']},
            **{f"{k}_{v}_Y": holdout_data[k][v]['Y'] for k in holdout_data for v in ['train', 'holdout']}
        )

    def load_split(self):
        """Load training and holdout splits from the .npz file."""
        loader = np.load(self.split_file)
        self.tables = {
            'FTER': {
                'train': {'X': loader['FTER_train_X'], 'Y': loader['FTER_train_Y']},
                'holdout': {'X': loader['FTER_holdout_X'], 'Y': loader['FTER_holdout_Y']}
            },
            'DTER': {
                'train': {'X': loader['DTER_train_X'], 'Y': loader['DTER_train_Y']},
                'holdout': {'X': loader['DTER_holdout_X'], 'Y': loader['DTER_holdout_Y']}
            },
            'FTMR': {
                'train': {'X': loader['FTMR_train_X'], 'Y': loader['FTMR_train_Y']},
                'holdout': {'X': loader['FTMR_holdout_X'], 'Y': loader['FTMR_holdout_Y']}
            },
            'DTMR': {
                'train': {'X': loader['DTMR_train_X'], 'Y': loader['DTMR_train_Y']},
                'holdout': {'X': loader['DTMR_holdout_X'], 'Y': loader['DTMR_holdout_Y']}
            },
            'TTER': {
                'train': {'X': loader['TTER_train_X'], 'Y': loader['TTER_train_Y']},
                'holdout': {'X': loader['TTER_holdout_X'], 'Y': loader['TTER_holdout_Y']}
            },
            'TTMR': {
                'train': {'X': loader['TTMR_train_X'], 'Y': loader['TTMR_train_Y']},
                'holdout': {'X': loader['TTMR_holdout_X'], 'Y': loader['TTMR_holdout_Y']}
            }
        }

    def define_pipelines(self, table_name):
        """Define pipelines and parameter grids for each model and approach."""
        alphas = np.logspace(-3, 3, 50)
        if table_name in ['TTER', 'TTMR']:
            pipelines = {
                "Lasso_TopK": Pipeline([
                    ("selector", TopKCorrelationSelector()),
                    ("scaler", StandardScaler()),
                    ("model", LassoCV(alphas=alphas, max_iter=10000, random_state=42))
                ]),
                "Lasso_PCA": Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(random_state=42)),
                    ("model", LassoCV(alphas=alphas, max_iter=10000, random_state=42))
                ]),
                "RF_TopK": Pipeline([
                    ("selector", TopKCorrelationSelector()),
                    ("model", RandomForestRegressor(random_state=42))
                ]),
                "RF_PCA": Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(random_state=42)),
                    ("model", RandomForestRegressor(random_state=42))
                ]),
                "PLS_TopK": Pipeline([
                    ("selector", TopKCorrelationSelector()),
                    ("pls", PLSRegression())
                ]),
                "PLS_PCA": Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(random_state=42)),
                    ("pls", PLSRegression())
                ])
            }
            param_grids = {
                "Lasso_TopK": {"selector__k": [10, 50, 100]},
                "Lasso_PCA": {"pca__n_components": [2, 5, 10,20]},
                "RF_TopK": {"selector__k": [10, 50, 100]},
                "RF_PCA": {"pca__n_components": [2, 5, 10,20]},
                "PLS_TopK": {"selector__k": [10, 50, 100], "pls__n_components": [2, 5, 10]},
                "PLS_PCA": {"pca__n_components": [2, 5, 10], "pls__n_components": [2, 5, 10]}
            }
        else:
            pipelines = {
                "Ridge_TopK": Pipeline([
                    ("selector", TopKCorrelationSelector()),
                    ("scaler", StandardScaler()),
                    ("model", RidgeCV(alphas=alphas))
                ]),
                "Ridge_PCA": Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(random_state=42)),
                    ("model", RidgeCV(alphas=alphas))
                ]),
                "RF_TopK": Pipeline([
                    ("selector", TopKCorrelationSelector()),
                    ("model", RandomForestRegressor(random_state=42))
                ]),
                "RF_PCA": Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(random_state=42)),
                    ("model", RandomForestRegressor(random_state=42))
                ]),
                "PLS_TopK": Pipeline([
                    ("selector", TopKCorrelationSelector()),
                    ("pls", PLSRegression())
                ]),
                "PLS_PCA": Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(random_state=42)),
                    ("pls", PLSRegression())
                ])
            }
            param_grids = {
                "Ridge_TopK": {"selector__k": [50, 100, 200]},
                "Ridge_PCA": {"pca__n_components": [10, 20, 50, 100]},
                "RF_TopK": {"selector__k": [50, 100, 200]},
                "RF_PCA": {"pca__n_components": [10, 20, 50, 100]},
                "PLS_TopK": {"selector__k": [50, 100, 200], "pls__n_components": [10, 20, 50]},
                "PLS_PCA": {"pca__n_components": [10, 20, 50, 100], "pls__n_components": [10, 20, 50]}
            }
        return pipelines, param_grids

    def evaluate_models(self, table_name, X_train, Y_train):
        pipelines, param_grids = self.define_pipelines(table_name)
        results = {}

        for name, pipeline in pipelines.items():
            logging.info(f"\n## {table_name} - {name}")
            print(f"Evaluating {name} for {table_name}")

            # Configure GridSearchCV to score both R² and MSE
            grid = GridSearchCV(
                pipeline,
                param_grids[name],
                cv=self.outer_cv,
                scoring={'r2': 'r2', 'mse': 'neg_mean_squared_error'},  # Compute both metrics
                refit='r2',  # Refit based on R²
                n_jobs=-1
            )
            grid.fit(X_train, Y_train)

            # Log results for all parameter combinations
            for params, mean_r2, std_r2, mean_mse, std_mse in zip(
                    grid.cv_results_['params'],
                    grid.cv_results_['mean_test_r2'],
                    grid.cv_results_['std_test_r2'],
                    grid.cv_results_['mean_test_mse'],
                    grid.cv_results_['std_test_mse']
            ):
                # Convert negative MSE to positive for readability
                mean_mse = -mean_mse
                # Std of negative MSE is already positive, no adjustment needed
                logging.info(
                    f"- Params: {params}, "
                    f"R2: {mean_r2:.15f}, Std R2: {std_r2:.15f}, "
                    f"MSE: {mean_mse:.15f}, Std MSE: {std_mse:.15f}"
                )

            # Log and store the best model
            best_pipeline = grid.best_estimator_
            results[name] = {
                "best_params": grid.best_params_,
                "best_r2": grid.best_score_,
                "pipeline": best_pipeline
            }
            logging.info(
                f"**Best {name}** - Params: {grid.best_params_}, "
                f"CV R2: {grid.best_score_:.15f}\n"
            )
            print(f"Best params: {grid.best_params_}, Best R2: {grid.best_score_:.15f}")

        return results

    def test_holdout(self, table_name, best_pipeline, X_train, Y_train, X_holdout, Y_holdout):
        """Test the best pipeline on the holdout set and log results."""
        best_pipeline.fit(X_train, Y_train)
        Y_pred = best_pipeline.predict(X_holdout)
        mse = mean_squared_error(Y_holdout, Y_pred)
        r2 = best_pipeline.score(X_holdout, Y_holdout)
        logging.info(f"{table_name} Holdout - MSE: {mse:.15f}, R2: {r2:.15f}")
        print(f"{table_name} Holdout - MSE: {mse:.15f}, R2: {r2:.15f}")
        return {"MSE": mse, "R2": r2}

    def run_analysis(self):
        """Run the full analysis across all tables."""
        results = {}
        for table_name in ['FTER', 'FTMR', 'DTER', 'DTMR', 'TTER', 'TTMR']:
            print(f"\n=== Processing {table_name} ===")
            logging.info(f"# {table_name}\n")
            X_train = self.tables[table_name]['train']['X']
            Y_train = self.tables[table_name]['train']['Y']
            X_holdout = self.tables[table_name]['holdout']['X']
            Y_holdout = self.tables[table_name]['holdout']['Y']

            # Evaluate models
            model_results = self.evaluate_models(table_name, X_train, Y_train)

            # Select best model based on CV R²
            best_model_name = max(model_results, key=lambda k: model_results[k]['best_r2'])
            best_pipeline = model_results[best_model_name]['pipeline']

            # Test on holdout set
            holdout_results = self.test_holdout(
                table_name, best_pipeline, X_train, Y_train, X_holdout, Y_holdout
            )

            # Store results
            results[table_name] = {
                "cv_results": model_results,
                "best_model": best_model_name,
                "holdout_results": holdout_results
            }
        return results


# Run the analysis

efs = EFS()
results = efs.run_analysis()
print("\nFinal Results:")
print(results)