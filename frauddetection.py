import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("~/Downloads/creditcard.csv")[:80_000]
X = df.drop(columns=['Time', 'Amount', 'Class']).values
y = df['Class'].values
print(f"Shapes of X={X.shape} y={y.shape}, #Fraud Cases={y.sum()}")

#scoring function
def min_recall_precision(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return min(recall, precision)

# Logistic Regression Grid Search
def logistic_regression_grid_search(X, y):
    param_grid = {'class_weight': [{0: 1, 1: v} for v in np.linspace(1, 20, 30)]}
    scoring = {
        'precision': make_scorer(precision_score), 
        'recall': make_scorer(recall_score),
        'min_both': make_scorer(min_recall_precision)
    }
    grid = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000),
        param_grid=param_grid,
        scoring=scoring,
        refit='min_both',
        return_train_score=True,
        cv=10,
        n_jobs=-1
    )
    grid.fit(X, y)
    return pd.DataFrame(grid.cv_results_)

# Isolation Forest Grid Search
def isolation_forest_grid_search(X, y):
    param_grid = {'contamination': np.linspace(0.001, 0.02, 10)}
    scoring = {
        'precision': make_scorer(lambda est, X, y: precision_score(y, np.where(est.predict(X) == -1, 1, 0))),
        'recall': make_scorer(lambda est, X, y: recall_score(y, np.where(est.predict(X) == -1, 1, 0)))
    }
    grid = GridSearchCV(
        estimator=IsolationForest(),
        param_grid=param_grid,
        scoring=scoring,
        refit='precision',
        cv=5,
        n_jobs=-1
    )
    grid.fit(X, y)
    return pd.DataFrame(grid.cv_results_)

# Plotting function 
def plot_grid_search_results(df_results, param_name, score_metrics):
    plt.figure(figsize=(12, 4))
    for score in score_metrics:
        plt.plot(
            df_results[param_name].apply(lambda x: x[1] if isinstance(x, dict) else x), 
            df_results[score], 
            label=score
        )
    plt.legend()
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(f'Grid Search Results for {param_name}')
    plt.show()

# Main 
if __name__ == "__main__":
    # Logistic Regression grid search and plotting
    df_logistic = logistic_regression_grid_search(X, y)
    plot_grid_search_results(df_logistic, 'param_class_weight', 
                             ['mean_test_recall', 'mean_test_precision', 'mean_test_min_both'])

    # Isolation Forest grid search and plotting
    df_isolation = isolation_forest_grid_search(X, y)
    plot_grid_search_results(df_isolation, 'param_contamination', 
                             ['mean_test_recall', 'mean_test_precision'])


