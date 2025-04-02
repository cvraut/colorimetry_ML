import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import logging


# Custom feature selector based on correlation
class TopKCorrelationSelector:
    def __init__(self, k=100):
        self.k = k
        self.selected_features = None

    def fit(self, X, y):
        corrs = np.array([abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])])
        self.selected_features = np.argsort(corrs)[-self.k:]
        return self

    def transform(self, X):
        return X[:, self.selected_features]


# Main class for Final Model Evaluation
class FinalModelEvaluation:
    def __init__(self, npz_file='../data/processed_data/all.npz'):
        """
        Initialize the FinalModelEvaluation class.

        Args:
            npz_file (str): Path to the NPZ file containing the data.
            fig_dir (str): Directory to save figures and results.
            log_file (str): Markdown file to log results.
        """
        self.npz_file = npz_file
        self.fig_dir = '../reports/figures/model_evaluation'
        self.split_file = '../data/processed_data/traintest/holdout_split.npz'
        os.makedirs(self.fig_dir, exist_ok=True)

        self.log_dir = '../reports/evaluation'
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = '../reports/evaluation/evaluation_results.md'
        self.outer_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

        # Set up logging to overwrite the MD file each run
        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                            format='%(message)s', filemode='w')
        logging.info("# Final Model Evaluation Results\n")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Load and split data
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
        """Define a specific pipeline for each table with manually set parameters."""
        if table_name == 'FTER':
            pipeline = Pipeline([
                ("selector", TopKCorrelationSelector(k=200)),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-3, 3, 50)))
            ])
        elif table_name == 'FTMR':
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=100, random_state=42)),
                ("pls", PLSRegression(n_components=10))
            ])
        elif table_name == 'DTER':
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=100, random_state=42)),
                ("pls", PLSRegression(n_components=10))
            ])
        elif table_name == 'DTMR':
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=100, random_state=42)),
                ("pls", PLSRegression(n_components=10))
            ])
        elif table_name == 'TTER':
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=5, random_state=42)),
                ("model", RandomForestRegressor( random_state=42))
            ])
        elif table_name == 'TTMR':
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=10, random_state=42)),
                ("pls", PLSRegression(n_components=10))
            ])
        else:
            raise ValueError(f"No pipeline defined for table: {table_name}")

        return pipeline

    def evaluate_models(self, table_name, X_train, Y_train, X_holdout, Y_holdout):
        """Evaluate the model using K-Fold CV and holdout set."""
        pipeline = self.define_pipelines(table_name)

        logging.info(f"\n## {table_name}")
        print(f"Evaluating model for {table_name}")

        # Define scorers for CV
        scorers = {
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error'
        }

        # Perform K-Fold CV
        cv_results = cross_validate(pipeline, X_train, Y_train, cv=self.outer_cv, scoring=scorers, n_jobs=-1)

        # Compute CV metrics
        cv_r2_mean = np.mean(cv_results['test_r2'])
        cv_r2_std = np.std(cv_results['test_r2'])
        cv_mse_mean = -np.mean(cv_results['test_neg_mse'])
        cv_mse_std = np.std(cv_results['test_neg_mse'])
        cv_mae_mean = -np.mean(cv_results['test_neg_mae'])
        cv_mae_std = np.std(cv_results['test_neg_mae'])
        cv_rmse_mean = np.sqrt(cv_mse_mean)

        logging.info(f"CV R2: {cv_r2_mean:.15f} ± {cv_r2_std:.15f}")
        logging.info(f"CV MSE: {cv_mse_mean:.15f} ± {cv_mse_std:.15f}")
        logging.info(f"CV MAE: {cv_mae_mean:.15f} ± {cv_mae_std:.15f}")
        logging.info(f"CV RMSE: {cv_rmse_mean:.15f}")

        # Train on full training set
        pipeline.fit(X_train, Y_train)

        # Evaluate on holdout set
        Y_pred = pipeline.predict(X_holdout)
        holdout_r2 = r2_score(Y_holdout, Y_pred)
        holdout_mse = mean_squared_error(Y_holdout, Y_pred)
        holdout_mae = mean_absolute_error(Y_holdout, Y_pred)
        holdout_rmse = np.sqrt(holdout_mse)
        tolerance = 0.001  # Assuming RIU units
        within_tolerance = np.mean(np.abs(Y_holdout - Y_pred) <= tolerance) * 100

        logging.info(f"Holdout R2: {holdout_r2:.15f}")
        logging.info(f"Holdout MSE: {holdout_mse:.15f}")
        logging.info(f"Holdout MAE: {holdout_mae:.15f}")
        logging.info(f"Holdout RMSE: {holdout_rmse:.15f}")
        logging.info(f"Percentage within ±0.001 RIU: {within_tolerance:.2f}%")

        # Compile results
        results = {
            "table_name": table_name,
            "pipeline_name": table_name,  # Assuming pipeline name is table_name for simplicity
            "parameters": pipeline.get_params(),
            "cv_r2_mean": cv_r2_mean,
            "cv_r2_std": cv_r2_std,
            "cv_mse_mean": cv_mse_mean,
            "cv_mse_std": cv_mse_std,
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
            "cv_rmse_mean": cv_rmse_mean,
            "holdout_r2": holdout_r2,
            "holdout_mse": holdout_mse,
            "holdout_mae": holdout_mae,
            "holdout_rmse": holdout_rmse,
            "within_tolerance": within_tolerance
        }

        return results, pipeline, Y_pred

    def save_holdout_to_md(self, table_name, pipeline, Y_holdout, Y_pred):
        """Save holdout test results to a Markdown file."""
        md_path = os.path.join(self.log_dir, f'{table_name}_holdout_results.md')
        with open(md_path, 'w') as f:
            f.write(f"# Holdout Test Results for {table_name}\n\n")
            f.write("## Pipeline Details\n")
            f.write(f"- **Pipeline Name**: {table_name}\n")
            f.write(f"- **Parameters**: {pipeline.get_params()}\n\n")
            f.write("## Holdout Predictions\n")
            f.write(" Actual,Predicted,Residual |\n")
            #f.write("|--------|-----------|----------|\n")
            for actual, pred in zip(Y_holdout, Y_pred):
                residual = actual - pred
                f.write(f"{actual:.8f},{pred:.8f},{residual:.8f} \n")

        logging.info(f"Holdout results saved to {md_path}")
        return md_path

    def plot_residuals(self, table_name, Y_holdout, Y_pred):
        """Plot residuals and save the figure."""
        residuals = Y_holdout - Y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(Y_holdout, residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel('Actual Values')
        plt.ylabel('Residuals')
        plt.title(f'Residuals for {table_name}')
        fig_path = os.path.join(self.fig_dir, f'{table_name}_residuals.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Residual plot saved to {fig_path}")
        return fig_path

    def save_results_to_csv(self, results_list):
        """Save evaluation results to a CSV file."""
        df = pd.DataFrame(results_list)
        csv_path = os.path.join(self.log_dir, 'evaluation_results.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"Results saved to {csv_path}")
        return csv_path

    def run_evaluation(self):
        """Run the evaluation for all tables."""
        results_list = []
        tables = ['FTER', 'FTMR', 'DTER', 'DTMR', 'TTER', 'TTMR']

        for table_name in tables:
            X_train = self.tables[table_name]['train']['X']
            Y_train = self.tables[table_name]['train']['Y']
            X_holdout = self.tables[table_name]['holdout']['X']
            Y_holdout = self.tables[table_name]['holdout']['Y']

            # Evaluate model
            results, pipeline, Y_pred = self.evaluate_models(table_name, X_train, Y_train, X_holdout, Y_holdout)

            # Save holdout results to MD
            md_path = self.save_holdout_to_md(table_name, pipeline, Y_holdout, Y_pred)
            results['holdout_md'] = md_path

            # Plot residuals
            fig_path = self.plot_residuals(table_name, Y_holdout, Y_pred)
            results['residual_plot'] = fig_path

            results_list.append(results)

        # Save all results to CSV
        csv_path = self.save_results_to_csv(results_list)
        return csv_path


# Example usage
if __name__ == "__main__":
    evaluator = FinalModelEvaluation()
    csv_path = evaluator.run_evaluation()
    print(f"Evaluation completed. Results saved to {csv_path}")