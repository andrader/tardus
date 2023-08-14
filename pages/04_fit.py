from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, get_scorer_names, get_scorer
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
    LeaveOneOut,
)
import pandas as pd
import streamlit as st
from sklearn.model_selection import cross_validate

state = st.session_state
import helpers as h


def assert_init(k):
    if not k in state:
        st.error(f"'{k}' not initialized")
        st.stop()


# def cross_val_score(estimator, X, y, cv, scoring):
#     scores = []
#     for train_idx, test_idx in cv.split(X, y):
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#         estimator.fit(X_train, y_train)
#         score = scoring(estimator, X_test, y_test)

#         scores.append(score)
#     return scores

assert_init("datasets")
assert_init("chosen_estimators")

st.selectbox("Dataset", [None] + list(state.datasets), key="fit_ds_name")

if state.fit_ds_name is not None:
    dataset: pd.DataFrame = state.datasets[state.fit_ds_name]

    cols = list(dataset.columns)
    target_idx = [i for i, c in enumerate(dataset.columns) if c.lower() == "target"]
    target_idx = target_idx[0] if target_idx else 0

    target = st.selectbox("Target", cols, target_idx, key="fit_target")

    available_features = [c for c in cols if c != target]
    features = st.multiselect(
        "Features", available_features, available_features, key="fit_features"
    )

    estimator_name = st.selectbox(
        "Estimators", state.chosen_estimators, key="fit_estimator"
    )
    estimator: BaseEstimator = state.chosen_estimators[estimator_name]["estimator"]

    cv_strategies = [None, KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit]
    cv_strategy = st.selectbox(
        "Cross validation",
        cv_strategies,
        format_func=lambda x: getattr(x, "__name__", str(x)),
    )

    # cv params inputs (exhaustive, all possible inputs) depending on strategy
    cv_params = {}
    if cv_strategy is not None:
        if cv_strategy == KFold:
            n_splits = st.number_input("Number of splits", 2, 10, 5)
            shuffle = st.checkbox("Shuffle", True)
            cv_params = {"n_splits": n_splits, "shuffle": shuffle}
        elif cv_strategy == StratifiedKFold:
            n_splits = st.number_input("Number of splits", 2, 10, 5)
            shuffle = st.checkbox("Shuffle", True)
            cv_params = {"n_splits": n_splits, "shuffle": shuffle}
        # Add more cases for other strategies

        cv_strategy = cv_strategy(**cv_params)

    # select metric/score (exhaustive list)
    metrics = get_scorer_names()
    metric = st.selectbox("Metric", metrics)
    scorer = get_scorer(metric)

    # st.write(str(type(estimator)))
    with st.expander("Data", expanded=False):
        st.dataframe(dataset[[target] + features], use_container_width=True, height=300)

    if st.button("Run cross validation", use_container_width=True):
        # get_data from dataset, features, target
        X = dataset[features]
        y = dataset[target]

        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # create a scorer based on the selected metric
        scoring = make_scorer(scorer)

        # cv on train data
        cv_results = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv_strategy,
            scoring=metric,
            error_score=True,
            return_train_score=True,
            n_jobs=-1,
        )

        state.cv_results = cv_results

    st.write("Results")
    if "cv_results" in state:
        # show cv results as dataframe.

        st.dataframe(
            pd.DataFrame(state.cv_results).sort_values("test_score"),
            use_container_width=True,
        )


## TODO: Grid Search
