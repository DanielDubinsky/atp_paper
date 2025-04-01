from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer
import pandas as pd

pd.set_option("display.float_format", lambda x: "%.3f" % x)
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from atp.preprocess import NormByCompound

import warnings

warnings.filterwarnings("ignore")


pathology_features = ["cytoplasmic_score", "nuclear_score", "contour_score"]
classic_features = ["pc", "curvature", "brightness"]
domains = ["A", "B", "C", "D"]


def get_metric_per_study(p, f=metrics.r2_score, split="test"):
    p = p.copy()[p.split == split]
    p["domain"] = p["domain"].map(
        {
            "A_cytation_v0": "C",
            "A_echo_rebel": "A",
            "B_echo_revolve_v0": "B",
            "A_thorlabs": "D",
        }
    )
    # p['study'] = p.apply(lambda x: f'{x.domain}_{x.study}', axis=1)
    return p.groupby(["domain"]).apply(lambda x: f(x.value, x.y_pred)).to_dict()


def regression_model_search(X, y, transforms):
    # Define the models and parameters
    models_params = {
        "DecisionTreeRegressor": {
            "model": Pipeline(
                [
                    ("transforms", transforms),
                    ("reg", DecisionTreeRegressor(random_state=42)),
                ]
            ),
            "params": {
                "reg__max_depth": [2, 3, 5],
                "reg__min_samples_split": [2, 4, 6],
                "reg__min_samples_leaf": [1, 2, 4],
            },
        },
        "GradientBoostingRegressor": {
            "model": Pipeline(
                [
                    ("transforms", transforms),
                    ("reg", GradientBoostingRegressor(random_state=42)),
                ]
            ),
            "params": {
                "reg__n_estimators": [10, 50, 100],
                "reg__learning_rate": [0.001, 0.01, 0.1, 1],
                "reg__max_depth": [2, 3, 5],
                "reg__min_samples_split": [2, 4, 6],
                "reg__min_samples_leaf": [1, 2, 4],
            },
        },
        "MLPRegressor": {
            "model": Pipeline(
                [("transforms", transforms), ("reg", MLPRegressor(random_state=42))]
            ),
            "params": {
                "reg__hidden_layer_sizes": [[8] * i for i in range(1, 3)]
                + [[16] * i for i in range(1, 3)]
                + [[32] * i for i in range(1, 3)],
                "reg__solver": ["adam"],
                "reg__early_stopping": [True, False],
                "reg__max_iter": [1000],
            },
        },
    }

    results = {}
    for model_name, mp in models_params.items():
        clf = GridSearchCV(
            mp["model"],
            mp["params"],
            cv=2,
            return_train_score=False,
            scoring="neg_mean_squared_error",
            n_jobs=8,
        )
        clf.fit(X, y)
        results[model_name] = {
            "best_score": clf.best_score_,
            "best_params": clf.best_params_,
            "clf": clf,
        }

    return results


def train(src_df, use_path_features, use_classic_features):
    df = pd.read_csv(src_df)
    df = NormByCompound()(df)

    transforms = []
    features = []
    if use_path_features:
        transforms += [
            (
                OneHotEncoder(
                    categories=[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
                    sparse_output=False,
                ),
                pathology_features,
            )
        ]
        features += pathology_features

    if use_classic_features:
        transforms += [(StandardScaler(), classic_features)]
        # transforms += [(FunctionTransformer(lambda x: x), classic_features)]
        # for c1, c2 in zip(classic_features, classic_control_features):
        #     df[c1] /= df[c2]
        features += classic_features
        # features += classic_control_features

    # select only rows that have requested features
    for f in features:
        df = df[df[f].notna()]

    print("training baseline")
    print(df.groupby(["split", "domain"]).count().iloc[:, 0])

    train = df.query('split == "train_val"')
    test = df.query('split == "test"')
    transforms = make_column_transformer(*transforms)
    X_train = train[features]
    y_train = train.value
    X_test = test[features]
    y_test = test.value

    # find hyper params
    tuned_params = regression_model_search(X_train, y_train, transforms)
    tuned_params = sorted(
        zip(tuned_params.keys(), tuned_params.values()),
        key=lambda x: x[1]["best_score"],
        reverse=True,
    )

    best_model_name = tuned_params[0][0]
    best_model_params = tuned_params[0][1]["best_params"]
    clf = tuned_params[0][1]["clf"].estimator

    print(best_model_name)
    print(best_model_params)

    rows = []
    for seed in range(42, 52):
        # clf = Pipeline([('transforms', transforms), ('reg', MLPRegressor())])
        clf.set_params(reg__random_state=seed, **best_model_params)
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        preds = df.copy()
        preds.loc[preds.split == "train_val", "y_pred"] = y_pred_train
        preds.loc[preds.split == "test", "y_pred"] = y_pred_test
        rows.append(
            {
                "seed": seed,
                "model": clf,
                "preds": preds,
                "r2_test": metrics.r2_score(y_test, y_pred_test),
                "r2_train": metrics.r2_score(y_train, y_pred_train),
                "mae_test": metrics.mean_absolute_error(y_test, y_pred_test),
            }
        )
    res_df = pd.DataFrame(rows)

    r2 = res_df.preds.map(lambda x: get_metric_per_study(x, metrics.r2_score))
    mae = res_df.preds.map(
        lambda x: get_metric_per_study(x, metrics.mean_absolute_error)
    )
    for d in domains:
        res_df[d + "_r2"] = r2.map(lambda x: x[d])
        res_df[d + "_mae"] = mae.map(lambda x: x[d])

    print(
        res_df[
            ["r2_train"] + [d + "_r2" for d in domains] + [d + "_mae" for d in domains]
        ].agg(["mean", "std"])
    )
    return res_df


src_df = "./data/data_labeled_w_classic_features.csv"
print("\t\t\t\tCVE baseline")
res_df = train(src_df, False, True)
res_df.to_pickle("./data/cve_baseline.pickle")

print("\t\t\t\tPathology baseline")
res_df = train(src_df, True, False)
res_df.to_pickle("./data/pathology_baseline.pickle")

print("\t\t\t\tPathology + CVE baseline")
res_df = train(src_df, True, True)
res_df.to_pickle("./data/pathology+cve_baseline.pickle")
