"""
scikit-learn estimators that count as models (Note 2)
=====================================================
Only predictive estimators are reported: classifiers, regressors, clustering
models and outlier detectors. Pre-processing and feature-extraction classes
(StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, SimpleImputer,
KNNImputer, IterativeImputer, CountVectorizer, TfidfVectorizer, DictVectorizer,
PCA, SelectKBest, ...), model-selection helpers (GridSearchCV, Pipeline) and
search structures (KDTree, BallTree, NearestNeighbors) are never in these lists.

The lists were generated from scikit-learn 1.9.1 with
`sklearn.utils.all_estimators(type_filter=...)` (outlier detectors: subclasses
of OutlierMixin; Gaussian mixtures added to clustering). Call
`regenerate()` to rebuild them from the installed scikit-learn version.
Older class names that no longer exist are kept in LEGACY so that projects
pinned to old versions are still recognised.
"""

from __future__ import annotations

from typing import Dict, Iterable, Set

SKLEARN_VERSION = "1.9.1"

CLASSIFIER = {
    "AdaBoostClassifier", "BaggingClassifier", "BernoulliNB", "CalibratedClassifierCV",
    "CategoricalNB", "ClassifierChain", "ComplementNB", "DecisionTreeClassifier",
    "DummyClassifier", "ExtraTreeClassifier", "ExtraTreesClassifier",
    "FixedThresholdClassifier", "GaussianNB", "GaussianProcessClassifier",
    "GradientBoostingClassifier", "HistGradientBoostingClassifier", "KNeighborsClassifier",
    "LabelPropagation", "LabelSpreading", "LinearDiscriminantAnalysis", "LinearSVC",
    "LogisticRegression", "LogisticRegressionCV", "MLPClassifier", "MultiOutputClassifier",
    "MultinomialNB", "NearestCentroid", "NuSVC", "OneVsOneClassifier", "OneVsRestClassifier",
    "OutputCodeClassifier", "PassiveAggressiveClassifier", "Perceptron",
    "QuadraticDiscriminantAnalysis", "RadiusNeighborsClassifier", "RandomForestClassifier",
    "RidgeClassifier", "RidgeClassifierCV", "SGDClassifier", "SVC", "SelfTrainingClassifier",
    "StackingClassifier", "TunedThresholdClassifierCV", "VotingClassifier",
}

REGRESSOR = {
    "ARDRegression", "AdaBoostRegressor", "BaggingRegressor", "BayesianRidge", "CCA",
    "DecisionTreeRegressor", "DummyRegressor", "ElasticNet", "ElasticNetCV",
    "ExtraTreeRegressor", "ExtraTreesRegressor", "GammaRegressor", "GaussianProcessRegressor",
    "GradientBoostingRegressor", "HistGradientBoostingRegressor", "HuberRegressor",
    "IsotonicRegression", "KNeighborsRegressor", "KernelRidge", "Lars", "LarsCV", "Lasso",
    "LassoCV", "LassoLars", "LassoLarsCV", "LassoLarsIC", "LinearRegression", "LinearSVR",
    "MLPRegressor", "MultiOutputRegressor", "MultiTaskElasticNet", "MultiTaskElasticNetCV",
    "MultiTaskLasso", "MultiTaskLassoCV", "NuSVR", "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV", "PLSCanonical", "PLSRegression",
    "PassiveAggressiveRegressor", "PoissonRegressor", "QuantileRegressor", "RANSACRegressor",
    "RadiusNeighborsRegressor", "RandomForestRegressor", "RegressorChain", "Ridge", "RidgeCV",
    "SGDRegressor", "SVR", "StackingRegressor", "TheilSenRegressor",
    "TransformedTargetRegressor", "TweedieRegressor", "VotingRegressor",
}

CLUSTER = {
    "AffinityPropagation", "AgglomerativeClustering", "BayesianGaussianMixture", "Birch",
    "BisectingKMeans", "DBSCAN", "GaussianMixture", "HDBSCAN", "KMeans", "MeanShift",
    "MiniBatchKMeans", "OPTICS", "SpectralClustering",
}

OUTLIER_DETECTOR = {
    "EllipticEnvelope", "IsolationForest", "LocalOutlierFactor", "OneClassSVM",
    "SGDOneClassSVM",
}

# removed / renamed in recent versions
LEGACY = {
    "classifier": {"RandomizedLogisticRegression", "LogisticRegressionCV"},
    "regressor": {"RandomizedLasso", "MultiTaskElasticNetCV"},
    "cluster": {"Ward", "MiniBatchKMeans"},
    "outlier_detector": set(),
}

FAMILIES: Dict[str, Set[str]] = {
    "classifier": CLASSIFIER,
    "regressor": REGRESSOR,
    "cluster": CLUSTER,
    "outlier_detector": OUTLIER_DETECTOR,
}


def sklearn_model_classes(types: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for t in types:
        out |= FAMILIES.get(t, set()) | LEGACY.get(t, set())
    return out


def regenerate() -> Dict[str, list]:  # pragma: no cover - helper for maintainers
    """Rebuild the lists from the installed scikit-learn (prints them)."""
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.base import OutlierMixin
    from sklearn.utils import all_estimators
    out = {t: sorted(n for n, _ in all_estimators(type_filter=t)) for t in ("classifier", "regressor", "cluster")}
    out["cluster"] = sorted((set(out["cluster"]) | {"GaussianMixture", "BayesianGaussianMixture"})
                            - {"FeatureAgglomeration"})
    out["outlier_detector"] = sorted(n for n, c in all_estimators() if issubclass(c, OutlierMixin))
    for k, v in out.items():
        print(k, v)
    return out
