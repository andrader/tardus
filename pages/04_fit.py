from sklearn.metrics import get_scorer_names
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


def assert_init(k):
    if k not in state:
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


def show_cross_validation(X, y):
    estimator_name = st.selectbox(
        "Estimators", state.chosen_estimators, key="fit_estimator"
    )
    estimator = state.chosen_estimators[estimator_name]["estimator"]

    cv_strategies = [KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit]
    cv_strategy = st.selectbox(
        "Cross validation",
        cv_strategies,
        format_func=lambda x: getattr(x, "__name__", str(x)),
    )

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

    if st.button("Run cross validation", use_container_width=True):
        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

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


if "data" in state and state.data:
    data = state.data
    dataset_name = data["dataset_name"]
    target_name = data["target_name"]
    feature_names = data["feature_names"]
    dataset = data["dataset"]
    features = data["features"]
    target = data["target"]

    with st.expander("Data", expanded=False):
        st.dataframe(dataset, use_container_width=True, height=300)

    # show_cross_validation(features, target)
