from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics

# models_params = {
#         'DecisionTree': {
#             'model': DecisionTreeClassifier(),
#             'params': {
#                 'criterion': ['gini', 'entropy'],
#                 'max_depth': [None, 2, 3, 5, 10, 20, 30, 40, 50],
#                 'min_samples_split': [2, 5, 10, 15],
#                 'min_samples_leaf': [1, 2, 4, 6, 8],
#                 'random_state': [42],
#                 'class_weight': [None, 'balanced'],
#                 'max_features': [None, 2, 3]
#             }
#         },
#         'RandomForest': {
#             'model': RandomForestClassifier(),
#             'params': {
#                 'n_estimators': [10, 50, 100, 200],
#                 'criterion': ['gini', 'entropy'],
#                 'max_depth': [None, 2, 3, 5, 10, 20, 30, 40, 50],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4],
#                 'random_state': [42],
#                 'class_weight': [None, 'balanced'],
#                 'max_features': [None, 2, 3]
#             }
#         },
#         'GradientBoosting': {
#             'model': GradientBoostingClassifier(),
#             'params': {
#                 'n_estimators': [10, 50, 100, 200],
#                 'learning_rate': [0.001, 0.01, 0.1, 1],
#                 'max_depth': [3, 10, 20, 30],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4],
#                 'random_state': [42],
#                 'max_features': [None, 2, 3]
#             }
#         }
#     }

models_params = {
        'DecisionTree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 3, 5, 7],  # Limited depth options to prevent overfitting
                'min_samples_split': [2, 4, 6],  # Considering the small dataset size
                'min_samples_leaf': [1, 2, 3],  # Small values due to limited data points
                'class_weight': [None, 'balanced'],
                'random_state': [42]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [10, 50, 100],  # A reasonable range considering dataset size
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 3, 5, 7],  # Similar to Decision Tree to avoid overfitting
                'min_samples_split': [2, 4, 6], 
                'min_samples_leaf': [1, 2, 3],
                'max_features': [None, 2, 4, 6],  # Considering the small number of features
                'class_weight': [None, 'balanced'],
                'random_state': [42]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(),
            'params': {
                'n_estimators': [10, 50, 100, 200],
                'learning_rate': [0.001, 0.01, 0.1, 1],
                'max_depth': [3, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'random_state': [42],
                'max_features': [None, 2, 3]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(),
            'params': {
                'class_weight': [None, 'balanced'],
                'max_iter': [2000]
            }
        }
    }

def tree_based_model_search(X, y, models=['DecisionTree', 'RandomForest', 'GradientBoosting'], cv=2, n_jobs=7):
    scorers = {
        'specificity': metrics.make_scorer(metrics.recall_score, pos_label=0),
        'sensitivity': metrics.make_scorer(metrics.recall_score),
        'roc_auc': metrics.make_scorer(metrics.roc_auc_score)
    }

    # Define the models and parameters
    results = {}
    for model_name in models:
        mp = models_params[model_name]
        clf = GridSearchCV(mp['model'], mp['params'], cv=cv, return_train_score=False, scoring=scorers, refit='roc_auc', n_jobs=n_jobs)
        clf.fit(X, y)
        results[model_name] = {
            'best_score': clf.best_score_,
            'best_params': clf.best_params_,
            'clf': clf
        }

    return results