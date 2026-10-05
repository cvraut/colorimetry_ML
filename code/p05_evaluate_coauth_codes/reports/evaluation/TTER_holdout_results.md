# Holdout Test Results for TTER

## Pipeline Details
- **Pipeline Name**: TTER
- **Parameters**: {'memory': None, 'steps': [('scaler', StandardScaler()), ('pca', PCA(n_components=5, random_state=42)), ('model', RandomForestRegressor(random_state=42))], 'transform_input': None, 'verbose': False, 'scaler': StandardScaler(), 'pca': PCA(n_components=5, random_state=42), 'model': RandomForestRegressor(random_state=42), 'scaler__copy': True, 'scaler__with_mean': True, 'scaler__with_std': True, 'pca__copy': True, 'pca__iterated_power': 'auto', 'pca__n_components': 5, 'pca__n_oversamples': 10, 'pca__power_iteration_normalizer': 'auto', 'pca__random_state': 42, 'pca__svd_solver': 'auto', 'pca__tol': 0.0, 'pca__whiten': False, 'model__bootstrap': True, 'model__ccp_alpha': 0.0, 'model__criterion': 'squared_error', 'model__max_depth': None, 'model__max_features': 1.0, 'model__max_leaf_nodes': None, 'model__max_samples': None, 'model__min_impurity_decrease': 0.0, 'model__min_samples_leaf': 1, 'model__min_samples_split': 2, 'model__min_weight_fraction_leaf': 0.0, 'model__monotonic_cst': None, 'model__n_estimators': 100, 'model__n_jobs': None, 'model__oob_score': False, 'model__random_state': 42, 'model__verbose': 0, 'model__warm_start': False}

## Holdout Predictions
Actual | Predicted | Residual 
--- | --- | --- 
85.00000000 | 84.18000000 | 0.82000000 
56.00000000 | 55.75000000 | 0.25000000 
67.00000000 | 66.37000000 | 0.63000000 
68.00000000 | 69.32000000 | -1.32000000 
46.00000000 | 47.22000000 | -1.22000000 
40.00000000 | 40.36000000 | -0.36000000 
23.00000000 | 23.49000000 | -0.49000000 
45.00000000 | 45.02000000 | -0.02000000 
11.00000000 | 11.21000000 | -0.21000000 
1.00000000 | 2.71000000 | -1.71000000 
19.00000000 | 19.60000000 | -0.60000000 
31.00000000 | 30.11000000 | 0.89000000 
98.00000000 | 97.66000000 | 0.34000000 
34.00000000 | 34.29000000 | -0.29000000 
78.00000000 | 76.89000000 | 1.11000000 
5.00000000 | 5.14000000 | -0.14000000 
94.00000000 | 93.73000000 | 0.27000000 
79.00000000 | 79.66000000 | -0.66000000 
13.00000000 | 13.39000000 | -0.39000000 
32.00000000 | 32.31000000 | -0.31000000 
