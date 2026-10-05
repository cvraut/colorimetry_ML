import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Explore:
    def __init__(self, npz_file='../data/processed_data/all.npz'):
        loader = np.load(npz_file)
        self.tables = {
            'FTER': loader['FTER'],
            'DTER': loader['DTER'],
            'FTMR': loader['FTMR'],
            'DTMR': loader['DTMR'],
            'TTER': loader['TTER'],
            'TTMR': loader['TTMR']
        }
        self.fig_dir = '../reports/figures/EDA'
        os.makedirs(self.fig_dir, exist_ok=True)
        self.md_file = '../reports/EDA.md'
        self.dtdir= '../reports/EDA/'
        with open(self.md_file, 'w') as f:
            f.write("# Exploratory Data Analysis\n\n")
        self.show()

    def append_to_md(self, section, content):
        with open(self.md_file, 'a') as f:
            f.write(f"## {section}\n\n{content}\n\n")

    def show(self):
        content = "**Overview**: Check shapes and infinities. First column is target (Y), rest are features (X).\n\n```plaintext\n"
        for table_name, data_array in self.tables.items():
            Y = data_array[:, 0]  # Target (first column)
            X = data_array[:, 1:]  # Features (remaining columns)
            content += (f"{table_name}:\n"
                       f"  Target (Y) shape: {Y.shape}\n"
                       f"  Features (X) shape: {X.shape}\n"
                       f"  Infinities in Y: {np.isinf(Y).sum()}\n"
                       f"  Infinities in X: {np.isinf(X).sum()}\n"
                       f"  X min: {np.min(X):.4f}, X max: {np.max(X):.4f}\n\n")
        content += "```"
        self.append_to_md("Show", content)
        print("Show results written to EDA.md")

    def column_minmax_explore(self):
        content = "**Min-Max Exploration**: Analyze feature column ranges.\n\n```plaintext\n"
        for table_name, data_array in self.tables.items():
            X = data_array[:, 1:]  # Features
            differences = np.max(X, axis=0) - np.min(X, axis=0)
            t1, t2, t3 = 0.05, 0.1, 0.15
            content += (f"{table_name}:\n"
                       f"  Min difference: {np.min(differences):.4f}, col: {np.argmin(differences)}\n"
                       f"  Max difference: {np.max(differences):.4f}, col: {np.argmax(differences)}\n"
                       f"  Top difference: {np.partition(differences, -1)[-1]:.4f}\n"
                       f"  <=0.05: {len(differences[differences <= t1])}\n"
                       f"  <=0.1: {len(differences[differences <= t2])}\n"
                       f"  <=0.15: {len(differences[differences <= t3])}\n\n")
            plt.scatter(range(len(differences)), differences, c='b', s=0.1)
            self.ext_data(differences, str(self.dtdir + table_name + "_minmax.csv"))
            plt.title(f"{table_name}: Max - Min Differences")
            plt.savefig(os.path.join(self.fig_dir, f"{table_name}_minmax.png"))
            plt.close()
        content += "```"
        self.append_to_md("Column Min-Max Explore", content)
        print("Min-Max results written to EDA.md")

    def zero_variance(self):
        content = "**Variance Analysis**: Examine feature variances with 10 ranges.\n\n```plaintext\n"
        for table_name, data_array in self.tables.items():
            X = data_array[:, 1:]  # Features
            variances = pd.DataFrame(X).var(axis=0)
            content += (f"{table_name}:\n"
                       f"  Min variance: {np.min(variances):.4f}, col: {np.argmin(variances)}\n"
                       f"  Max variance: {np.max(variances):.4f}, col: {np.argmax(variances)}\n"
                       f"  Top 5 variances: {np.partition(variances, -5)[-5:].tolist()}\n")
            # 10 ranges for variance
            var_min, var_max = np.min(variances), np.max(variances)
            bins = np.linspace(var_min, var_max, 11)
            counts, _ = np.histogram(variances, bins=bins)
            content += "  Variance ranges:\n"
            for i in range(len(counts)):
                content += f"    {bins[i]:.4f}–{bins[i+1]:.4f}: {counts[i]}\n"
            content += "\n"
            plt.scatter(range(len(variances)), variances, c='b', s=0.1)
            self.ext_data(list(variances), str(self.dtdir + table_name + "_variance.csv"))
            plt.title(f"{table_name}: Variance")
            plt.savefig(os.path.join(self.fig_dir, f"{table_name}_variance.png"))
            plt.close()
        content += "```"
        self.append_to_md("Zero Variance", content)
        print("Variance results written to EDA.md")

    def mean_std(self):
        content = "**Mean and Std**: Feature means and standard deviations.\n\n```plaintext\n"
        for table_name, data_array in self.tables.items():
            X = data_array[:, 1:]  # Features
            means = np.mean(X, axis=0)
            stds = np.std(X, axis=0)
            content += (f"{table_name}:\n"
                       f"  Max mean: {np.max(means):.4f}, col: {np.argmax(means)}\n"
                       f"  Min mean: {np.min(means):.4f}, col: {np.argmin(means)}\n"
                       f"  Max std: {np.max(stds):.4f}, col: {np.argmax(stds)}\n"
                       f"  Min std: {np.min(stds):.4f}, col: {np.argmin(stds)}\n\n")
            plt.scatter(range(len(means)), means, c='b', s=0.1)
            self.ext_data(means, str(self.dtdir + table_name + "_means.csv"))
            plt.title(f"{table_name}: Means")
            plt.savefig(os.path.join(self.fig_dir, f"{table_name}_means.png"))
            plt.close()
            plt.scatter(range(len(stds)), stds, c='b', s=0.1)
            self.ext_data(stds, str(self.dtdir + table_name + "_stds.csv"))
            plt.title(f"{table_name}: Std Devs")
            plt.savefig(os.path.join(self.fig_dir, f"{table_name}_stds.png"))
            plt.close()
        content += "```"
        self.append_to_md("Mean and Std", content)
        print("Mean/Std results written to EDA.md")

    def target_correlation(self):
        content = "**Target Correlation**: Pearson and Spearman correlations with target, 10 ranges.\n\n```plaintext\n"
        for table_name, data_array in self.tables.items():
            Y = data_array[:, 0]  # Target
            X = data_array[:, 1:]  # Features
            pearson_corrs = []
            spearman_corrs = []
            for i in range(X.shape[1]):
                p_r, _ = pearsonr(X[:, i], Y)
                s_r, _ = spearmanr(X[:, i], Y)
                pearson_corrs.append(abs(p_r))
                spearman_corrs.append(abs(s_r))
            p_sorted_idx = np.argsort(pearson_corrs)[::-1][:10]
            s_sorted_idx = np.argsort(spearman_corrs)[::-1][:10]
            content += (f"{table_name}:\n"
                       f"  Top 10 Pearson: {p_sorted_idx.tolist()}\n"
                       f"  Pearson values: {[pearson_corrs[i] for i in p_sorted_idx]}\n"
                       f"  Top 10 Spearman: {s_sorted_idx.tolist()}\n"
                       f"  Spearman values: {[spearman_corrs[i] for i in s_sorted_idx]}\n")
            # 10 ranges for correlations (0 to 1, since abs values)
            bins = np.linspace(0, 1, 11)
            p_counts, _ = np.histogram(pearson_corrs, bins=bins)
            s_counts, _ = np.histogram(spearman_corrs, bins=bins)
            content += "  Pearson correlation ranges:\n"
            for i in range(len(p_counts)):
                content += f"    {bins[i]:.1f}–{bins[i+1]:.1f}: {p_counts[i]}\n"
            content += "  Spearman correlation ranges:\n"
            for i in range(len(s_counts)):
                content += f"    {bins[i]:.1f}–{bins[i+1]:.1f}: {s_counts[i]}\n"
            content += "\n"
            plt.scatter(range(len(pearson_corrs)), pearson_corrs, c='b', s=0.1, label='Pearson')
            self.ext_data(list(pearson_corrs), str(self.dtdir + table_name + "_pearsoncorr.csv"))
            plt.scatter(range(len(spearman_corrs)), spearman_corrs, c='r', s=0.1, alpha=0.5, label='Spearman')
            self.ext_data(list(spearman_corrs), str(self.dtdir + table_name + "_spearmancorrs.csv"))
            plt.legend()
            plt.title(f"{table_name}: Target Correlation")
            plt.savefig(os.path.join(self.fig_dir, f"{table_name}_correlation.png"))
            plt.close()
        content += "```"
        self.append_to_md("Target Correlation", content)
        print("Correlation results written to EDA.md")

    def pca_test(self):
        content = "**PCA Test**: Variance explained by components.\n\n```plaintext\n"
        for table_name, data_array in self.tables.items():
            X = data_array[:, 1:]  # Features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(random_state=42)
            pca.fit(X_scaled)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            components_90 = np.argmax(cumulative_variance >= 0.90) + 1
            components_95 = np.argmax(cumulative_variance >= 0.95) + 1
            components_99 = np.argmax(cumulative_variance >= 0.99) + 1
            content += (f"{table_name}:\n"
                       f"  Components for 90% variance: {components_90}\n"
                       f"  Components for 95% variance: {components_95}\n"
                       f"  Components for 99% variance: {components_99}\n\n")
            plt.scatter(range(len(cumulative_variance)), cumulative_variance, c='b', s=0.5)
            self.ext_data(list(cumulative_variance), str(self.dtdir + table_name + "_cummulativevariance.csv"))
            plt.title(f"{table_name}: PCA Cumulative Variance")
            plt.savefig(os.path.join(self.fig_dir, f"{table_name}_pca.png"))
            plt.close()
        content += "```"
        self.append_to_md("PCA Test", content)
        print("PCA results written to EDA.md")

    def ext_data(self,data,file_path):
        """
            Save data (list or NumPy array) to a CSV file at the specified path.

            Parameters:
            data : list or numpy.ndarray
                The data to be saved (e.g., correlation coefficients)
            path : str
                The file path (including filename) where the CSV will be saved
            """
        try:
            # Convert list to NumPy array if necessary
            if isinstance(data, list):
                data = np.array(data)
            elif not isinstance(data, np.ndarray):
                raise TypeError("Data must be a list or NumPy array")

            # Save the array to CSV
            np.savetxt(file_path, data, delimiter=',', fmt='%.15f')  # Adjust precision as needed
            print(f"Data successfully saved to {file_path}")

        except Exception as e:
            print(f"Error saving data to CSV: {str(e)}")
# Run analysis
e = Explore()
e.column_minmax_explore()
e.zero_variance()
e.mean_std()
e.target_correlation()
e.pca_test()