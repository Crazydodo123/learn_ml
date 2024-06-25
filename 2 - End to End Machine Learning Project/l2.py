from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()


from sklearn.model_selection import train_test_split

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0, 1.5, 3, 4.5, 6, np.inf],
                               labels=[1, 2, 3, 4, 5])

strat_train_set, strat_test_set = train_test_split(
housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self     # fit always return self
    
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
    

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                            "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object))
    ], remainder=default_num_pipeline)

housing_prepared = preprocessing.fit_transform(housing)


housing_prepared_fr = pd.DataFrame(
    housing_prepared,
    columns=preprocessing.get_feature_names_out(),
    index=housing.index)

# from sklearn.linear_model import LinearRegression

# lin_reg = make_pipeline(preprocessing, LinearRegression())
# lin_reg.fit(housing, housing_labels)

# housing_predictions = lin_reg.predict(housing)

# from sklearn.metrics import mean_squared_error

# lin_rmse = mean_squared_error(housing_labels, housing_predictions,
#                               squared=False)
# print(lin_rmse)

# from sklearn.tree import DecisionTreeRegressor

# tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
# tree_reg.fit(housing, housing_labels)

# housing_predictions = tree_reg.predict(housing)
# tree_rmse = mean_squared_error(housing_labels, housing_predictions,
#                                squared=False)
# print(tree_rmse)

from sklearn.model_selection import cross_val_score

# tree_rsmes = -cross_val_score(tree_reg, housing, housing_labels,
#                               scoring="neg_root_mean_squared_error", cv=10)

# print(pd.Series(tree_rsmes).describe())

# lin_rsmes = -cross_val_score(lin_reg, housing, housing_labels,
#                               scoring="neg_root_mean_squared_error", cv=10)

# print(pd.Series(lin_rsmes).describe())

# from sklearn.ensemble import RandomForestRegressor
# forest_reg = make_pipeline(preprocessing,
#                            RandomForestRegressor(random_state=42))
# forest_rsmes = -cross_val_score(forest_reg, housing, housing_labels,
#                               scoring="neg_root_mean_squared_error", cv=10,
#                               n_jobs=-1)

# print(pd.Series(forest_rsmes).describe())


# from sklearn.ensemble import ExtraTreesRegressor
# print('extra_trees')
# x_trees_reg = make_pipeline(preprocessing,
#                            ExtraTreesRegressor(random_state=42))
# print('extra_trees cross_val')
# x_trees_reg = -cross_val_score(x_trees_reg, housing, housing_labels,
#                               scoring="neg_root_mean_squared_error", cv=10,
#                               n_jobs=-1)

# print(pd.Series(x_trees_reg).describe())


from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42))
])
# param_grid = [
#     {"preprocessing__geo__n_clusters": [5, 8, 10],
#      "random_forest__max_features": [4, 6, 8]},
#     {"preprocessing__geo__n_clusters": [10, 15],
#      "random_forest__max_features": [6, 8, 10]},
# ]

# grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
#                            scoring="neg_root_mean_squared_error")
# grid_search.fit(housing, housing_labels)

# cv_res = pd.DataFrame(grid_search.cv_results_)
# cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)

# cv_res = cv_res[["param_preprocessing__geo__n_clusters",
#                  "param_random_forest__max_features", "split0_test_score",
#                  "split1_test_score", "split2_test_score", "mean_test_score"]]
# score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
# cv_res.columns = ["n_clusters", "max_features"] + score_cols
# cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

# print(cv_res.head())

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    "preprocessing__geo__n_clusters": randint(low=3, high=50),
    "random_forest__max_features": randint(low=2, high=20)
}

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring="neg_root_mean_squared_error", random_state=42
)

rnd_search.fit(housing, housing_labels)