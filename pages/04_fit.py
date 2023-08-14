from sklearn.base import BaseEstimator
import streamlit as st

state = st.session_state
import helpers as h


def assert_init(k):
    if not k in state:
        st.error(f"'{k}' not initialized")
        st.stop()


assert_init("datasets")
assert_init("chosen_estimators")
import pandas as pd

st.selectbox("Dataset", [None] + list(state.datasets), key="fit_ds_name")

if state.fit_ds_name is not None:
    dataset: pd.DataFrame = state.datasets[state.fit_ds_name]

    # target = list()#[dataset.columns.str.match("target", case=False)]
    cols = list(dataset.columns)
    target_idx = [i for i, c in enumerate(dataset.columns) if c.lower() == "target"]
    target_idx = target_idx[0] if target_idx else 0
    st.write(target_idx)

    target = st.selectbox("Target", cols, target_idx, key="fit_target")

    available_features = [c for c in cols if c != target]
    features = st.multiselect(
        "Features", available_features, available_features, key="fit_features"
    )

    estimator_name = st.selectbox(
        "Estimators", state.chosen_estimators, key="fit_estimator"
    )
    estimator: BaseEstimator = state.chosen_estimators[estimator_name]

    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        TimeSeriesSplit,
        train_test_split,
        LeaveOneOut,
    )

    cv_strategies = [None, KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit]
    cv_strategy = st.selectbox(
        "Cross validation",
        cv_strategies,
        format_func=lambda x: getattr(x, "__name__", str(x)),
    )

    # select metric/score
    # [IMPLEMENT]

    # cv params input depending on strategy
    # [IMPLEMENT]

    cv_strategy = cv_strategy(cv_params)

    if st.button("Fit"):
        X, y = get_data(dataset, features, target)  # [IMPLEMENT]

        # split train and test data
        # [IMPLEMENT]

        # cv on train data
        fit_estimator(X, y, estimator, cv_strategy)  # [IMPLEMENT]

        # any other step?
        # [IMPLEMENT]

    st.write("Results")
    # show cv results as dataframe.
    # [IMPLEMENT]
