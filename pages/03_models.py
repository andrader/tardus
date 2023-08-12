import time
import typing
from numbers import Integral, Number

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import (
    LinearRegression,
    QuantileRegressor,
    RANSACRegressor,
    HuberRegressor,
    TheilSenRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.validation import check_is_fitted
from streamlit import session_state as state
from streamlit.runtime.state import NoValue
from sklearn.utils._param_validation import Interval, Options
import inspect
import helpers as h

# 1. clip quantiles => LinearRegression
# 1. winsorize => LinearRegression
# 1. clip studentized residuals => LinearRegression

# https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
# compare with: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.outlier_test.html


def get_studentized_residuals(X, y):
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    mean_X = np.mean(X)
    mean_Y = np.mean(Y)

    n = len(X)

    diff_mean_sqr = np.dot((X - mean_X), (X - mean_X))
    beta1 = np.dot((X - mean_X), (Y - mean_Y)) / diff_mean_sqr
    beta0 = mean_Y - beta1 * mean_X

    y_hat = beta0 + beta1 * X
    residuals = Y - y_hat

    sigma2 = (X - mean_X) ** 2 / diff_mean_sqr + (1 / n)
    Var_e = np.sqrt(np.sum((Y - y_hat) ** 2) / (n - 2))
    SE_regression = Var_e * ((1.0 / n) + (X - mean_X) ** 2 / diff_mean_sqr) ** 0.5

    studentized_residuals = residuals / SE_regression

    return studentized_residuals


def quantile_y_outliers(y, q=0.05):
    lower_bound = np.quantile(y, q)
    upper_bound = np.quantile(y, 1 - q)
    outliers = np.logical_or(y < lower_bound, y > upper_bound)

    return dict(
        outliers=outliers, lower_bound=lower_bound, upper_bound=upper_bound, q=q
    )


class FilteredLinearRegression(LinearRegression):
    def __init__(self, q=0.05):
        super().__init__()
        self.q_ = q
        self.func_ = quantile_y_outliers
        self.kwargs_ = dict(q=self.q_)

    def fit(self, X, y):
        res = self.func_(X, y, **self.kwargs_)
        for k, v in res.items():
            if k.endswith("_"):
                setattr(self, k, v)

        msk = ~res["outliers"]
        X_filtered = X[msk]
        y_filtered = y[msk]

        return super().fit(X_filtered, y_filtered)


def inject_before_fit(estimator: BaseEstimator, func, **kwargs) -> BaseEstimator:
    # add attributes to the estimator
    for k, v in kwargs.items():
        setattr(estimator, k, v)

    # call func to filter X, y before fitting
    def fit(estimator, X, y):
        res = func(X, y, **kwargs)

        assert "outliers" in res, "func must return dict with key 'outliers'"

        # add attributes to the object
        for k, v in res.items():
            if k.endswith("_"):
                setattr(estimator, k, v)

        msk = ~res["outliers"]
        X_filtered = X[msk]
        y_filtered = y[msk]

        print("Calling fit")
        return estimator.fit(X=X_filtered, y=y_filtered)

    estimator.fit = fit.__get__(estimator)

    return estimator


ESTIMATORS = {
    "OLS": LinearRegression(),
    "RANSAC": RANSACRegressor(min_samples=50, residual_threshold=5.0),
    # non-parametric, but doesn't scale well
    "TheilSenRegressor": TheilSenRegressor(),
    "HuberRegressor": HuberRegressor(),  # epsilon=1.35
    "QuantileRegressor": QuantileRegressor(solver="highs"),
    "FilteredLinearRegression": FilteredLinearRegression(q=0.01),
}


def create_param_input(constraint, default, name, key=None, disabled=False):
    val = default

    if constraint == "random_state":
        constraint = Interval(Integral, 0, right=None, closed="left")

    if isinstance(constraint, Interval) and isinstance(constraint.left, Integral):
        convert = int
        min_value = (
            None
            if constraint.left in (None, float("-inf"), float("-inf"))
            else convert(constraint.left)
        )
        max_value = (
            None
            if constraint.right in (None, float("inf"), float("inf"))
            else convert(constraint.right)
        )

        value = (
            NoValue()
            if default in (None, float("-inf"), float("inf"))
            else convert(default)
        )

        val = st.number_input(
            label=str(constraint),
            min_value=min_value,
            max_value=max_value,
            value=value,
            key=key,
            label_visibility="hidden",
            disabled=disabled,
        )

    elif isinstance(constraint, list):
        options = list(constraint.options)
        val = st.selectbox(
            label="",
            options=options,
            index=options.index(default) if default in options else 0,
            key=key,
            label_visibility="hidden",
            disabled=disabled,
        )

    elif constraint == "boolean":
        val = st.checkbox(
            label="",
            value=default,
            key=key,
        )

    elif isinstance(constraint, Number) or (
        isinstance(constraint, type) and issubclass(constraint, Number)
    ):
        converter = int if constraint is Integral else float

        val = st.number_input(
            label="",
            value=default,
            key=key,
            label_visibility="hidden",
            disabled=disabled,
        )

        val = converter(val)

    else:
        st.write(f"Input not supported: {str(constraint)} ({str(type(constraint))})")

    return val


class NoDefault:
    pass


def get_params(estimator: BaseEstimator) -> dict[str, tuple[list, NoDefault]]:
    params_names = estimator.get_params().keys()
    sig = inspect.signature(estimator.__init__)
    defaults = {k: v.default for k, v in sig.parameters.items() if k in params_names}

    dict_constraints = estimator._parameter_constraints
    params = {
        k: (dict_constraints.get(k, [None]), defaults[k])
        for k in estimator._get_param_names()
    }

    return params


def add_estimator():
    # Select estimator, then input params for that estimator

    container = st.container()

    with container:
        st.header("Models")

        select_estimator = st.selectbox("Select estimator", options=ESTIMATORS)

        estimator = ESTIMATORS[select_estimator]

        # add docs
        add_docs(estimator)

        # multiselect for params
        params = get_params(estimator)
        select_params = st.multiselect("Select parameters", options=list(params.keys()))

        label = st.text_input("Label", value=select_estimator)

        params_values = create_params_widgets(
            select_estimator=select_estimator,
            params=params,
            select_params=select_params,
        )

        estimator_class = (
            estimator if isinstance(estimator, type) else estimator.__class__
        )
        result = (estimator_class, params_values)

        if label in state.chosen_estimators:
            st.warning(
                f"Label '{label}' already exists in chosen estimators. It will be overwritten."
            )

        with container:
            cols = st.columns([1, 1])
            with cols[0]:
                st.button(
                    "Add",
                    use_container_width=True,
                    on_click=lambda: state.chosen_estimators.update({label: result}),
                )
            with cols[1]:
                st.button(
                    "Delete",
                    use_container_width=True,
                    disabled=label not in state.chosen_estimators,
                    on_click=lambda: state.chosen_estimators.pop(label, None),
                )

        st.write("Parameters values:", params_values)


def create_params_widgets(select_estimator, params, select_params):
    params_values = {}

    for name in select_params:
        with st.expander(f"**{name}**", expanded=True):
            cols = st.columns([1, 1])

            with cols[0]:
                use_default = st.checkbox(
                    f"Use default: {params[name][1]}",
                    value=False,
                    key=name + " use default",
                )

            with cols[1]:
                constraints = params[name][0]
                if len(constraints) == 1:
                    constraint = constraints[0]
                else:
                    constraint = st.selectbox(
                        "",
                        options=constraints,
                        index=constraints.index(params[name][1])
                        if params[name][1] in constraints
                        else 0,
                        key=" ".join([select_estimator, name, "param"]),
                        label_visibility="hidden",
                    )

                val = create_param_input(
                    constraint=constraint,
                    default=params[name][1],
                    name=name,
                    key=" ".join([select_estimator, name]),
                    disabled=use_default,
                )

                params_values[name] = val

    # overwrite defaults with user input
    defaults = {k: v[1] for k, v in params.items()}
    params_values = dict(defaults, **params_values)

    return params_values


def add_docs(estimator):
    module = estimator.__module__.split(".", 1)[0]
    if "sklearn" in module:
        name = f"{module}.regression.{estimator.__class__.__name__}"
        link = f"https://scikit-learn.org/stable/modules/generated/{name}.html"
        st.markdown(f"Docs: [{name}]({link})")


# delete estimator
if "chosen_estimators" not in state:
    state["chosen_estimators"] = dict()

cols = st.columns([3, 4])
with cols[0]:
    add_estimator()

with cols[1]:
    st.subheader("Chosen estimators")
    st.button(
        "Clear all estimators",
        use_container_width=True,
        on_click=lambda: state.chosen_estimators.clear(),
    )
    with st.expander("See chosen"):
        for label, (cls, params) in state.chosen_estimators.items():
            st.write(f"## {label} ({cls.__name__})")
            st.write("Parameters:")
            for k, v in params.items():
                st.write(f"- {k}: {v}")
            st.button("Add")
            state.chosen_estimators[label] = (cls, params)
