from numbers import Integral, Number

import pandas as pd
from sklearn import clone
import streamlit as st
from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    LinearRegression,
    QuantileRegressor,
    RANSACRegressor,
    HuberRegressor,
    TheilSenRegressor,
)
from streamlit import session_state as state
from streamlit.runtime.state import NoValue
from sklearn.utils._param_validation import Interval, Options, Hidden
import inspect
from typing import Any


ESTIMATORS = {
    "OLS": LinearRegression,
    "RANSAC": RANSACRegressor(min_samples=50, residual_threshold=5.0),
    # non-parametric, but doesn't scale well
    "TheilSenRegressor": TheilSenRegressor,
    "HuberRegressor": HuberRegressor(),  # epsilon=1.35
    "QuantileRegressor": QuantileRegressor(solver="highs"),
}


def init(k, v, d=state):
    if k not in d:
        d[k] = v


def get_grid(nobs, ncols=2):
    for i in range(nobs):
        col = i % ncols
        if col == 0:
            cols = st.columns(ncols)
        yield cols[col]


def get_params_constraints(estimator: BaseEstimator) -> dict[str, tuple[list, Any]]:
    params = estimator.get_params()
    constraints = estimator._parameter_constraints

    # sigparams = inspect.signature(estimator.__init__).parameters
    # defaults = {k: sigparams[k].default for k in params}
    # # overwrite signature defaults with instance params
    # params = dict(defaults, **params)

    # estimator_kwargs = estimator.get_params()
    # params = func_sig.bind(**estimator_kwargs)
    # params.apply_defaults()

    params = {k: (constraints.get(k), params[k]) for k in params}

    return params


class _NoVal(NoValue):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


def get_widget_espec(constr, default):
    # """
    # A parameter is valid if it satisfies one of the constraints from the list.
    #     Constraints can be:
    #     - an Interval object, representing a continuous or discrete range of numbers
    #     - the string "array-like"
    #     - the string "sparse matrix"
    #     - the string "random_state"
    #     - callable
    #     - None, meaning that None is a valid value for the parameter
    #     - any type, meaning that any instance of this type is valid
    #     - an Options object, representing a set of elements of a given type
    #     - a StrOptions object, representing a set of strings
    #     - the string "boolean"
    #     - the string "verbose"
    #     - the string "cv_object"
    #     - a MissingValues object representing markers for missing values
    #     - a HasMethods object, representing method(s) an object must have
    #     - a Hidden object, representing a constraint not meant to be exposed to the user
    # """
    def _none_widget(**kwargs):
        st.text_input(value=f"{str(default)}", **kwargs, disabled=True)
        return

    widget = _none_widget
    widget_kwargs = dict()

    if constr == "random_state":
        constr = Interval(Integral, 0, right=None, closed="left")
    # if isinstance(constr, Hidden):
    #     constr = constr.constraint

    if constr is None:
        pass
    elif isinstance(constr, Interval):
        convert = int if constr.type is Integral else float
        invalids = (None, float("-inf"), float("inf"))

        min_value = None if constr.left in invalids else convert(constr.left)
        max_value = None if constr.right in invalids else convert(constr.right)
        value = _NoVal() if default in invalids else convert(default)

        if value is not _NoVal():
            if (min_value is not None and value < min_value) or (
                max_value is not None and value > max_value
            ):
                value = _NoVal()

        widget = st.number_input
        widget_kwargs = dict(
            min_value=min_value,
            max_value=max_value,
            value=value,
        )

    elif isinstance(constr, Options):
        options = list(constr.options)
        widget = st.selectbox
        widget_kwargs = dict(
            options=options,
            index=options.index(default) if default in options else 0,
        )

    elif constr == "boolean":
        widget = st.checkbox
        widget_kwargs = dict(value=default, 
                            #  label_visibility="visible",
                             )

    elif isinstance(constr, Number) or (
        isinstance(constr, type) and issubclass(constr, Number)
    ):
        converter = int if constr is Integral else float
        widget = st.number_input
        widget_kwargs = dict(
            value=converter(0 if default is None else default),
        )

    else:

        def foo(**kwargs):
            st.write(f"Not supported: {str(constr)}")
            return default

        widget = foo

    return widget, widget_kwargs


def callback_add_estimator():
    params = state.params_values
    estimator = state._estimator.__class__().set_params(**params)

    state.chosen_estimators[state.estimator_label] = dict(
        estimator=estimator,
        # params=params
    )
    st.toast("Added")


def callback_update_use_default(default, key_usedefault, key_value):
    value = state.get(key_value, default)
    state[key_usedefault] = False#(value is default) or (value == default)


def create_params_widgets(estimator_name, params, select_params):
    params_values = {}

    containers = list(get_grid(len(select_params), 3))
    for (name, (constraints, default)), container in zip(params.items(), containers):
        if name not in select_params:
            continue

        key_usedefault = "_".join([estimator_name, name, "usedefault"])
        key_value = "_".join([estimator_name, name, "value"])
        constraints = [c for c in constraints if c is not None]

        with container:
            with st.expander(f"**{name}**", expanded=True):
                if len(constraints) == 1:
                    constr = constraints[0]
                else:
                    index = constraints.index(default) if default in constraints else 0

                    constraint_key = " ".join([estimator_name, name, "constraint"])
                    constr = st.selectbox(
                        label=constraint_key,
                        options=list(constraints),
                        index=index,
                        key=constraint_key,
                        label_visibility="hidden",
                    )

                widget, widget_kwargs = get_widget_espec(constr=constr, default=default)
                

                _kwargs = dict(
                    label_visibility="hidden",
                )
                _kwargs.update(widget_kwargs)

                val = widget(
                    label=str(name),
                    **_kwargs,
                    key=key_value,
                    on_change=callback_update_use_default,
                    args=(default, key_usedefault, key_value),
                )

                use_default = st.checkbox(
                    f"Use default `{default}`",
                    value=True,
                    key=key_usedefault,
                    help=f"Ignore input value and use the default `{default}`"
                )

                params_values[name] = default if use_default else val

    # overwrite defaults with user input
    defaults = {k: v[1] for k, v in params.items()}
    params_values = dict(defaults, **params_values)

    return params_values


def show_params_inputs():
    select_estimator = state.select_estimator
    estimator = state._estimator

    params = get_params_constraints(estimator)

    select_params = st.multiselect(
        "Select parameters", options=list(params.keys()), default=list(params.keys()),
        help="Select model parameters"
    )

    params_values = create_params_widgets(
        estimator_name=select_estimator,
        params=params,
        select_params=select_params,
    )

    state.params_values = params_values


def show_add_model():
    st.subheader("Add model")
    cols = st.columns(2)

    with cols[0]:
        select_estimator = st.selectbox(
            "Select model class",
            options=ESTIMATORS,
            key="select_estimator",
        )
    _estimator = ESTIMATORS[state.select_estimator]
    _estimator = _estimator if isinstance(_estimator, BaseEstimator) else _estimator()
    state._estimator = _estimator

    with cols[1]:
        st.text_input("Label", select_estimator, key="estimator_label")

    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.button("Add", use_container_width=True, on_click=callback_add_estimator)

    with cols[1]:
        st.button(
            "Delete",
            use_container_width=True,
            disabled=state.estimator_label not in state.chosen_estimators,
            on_click=lambda: state.chosen_estimators.pop(state.estimator_label, None),
        )

    with cols[2]:
        st.button(
            "Delete ALL models",
            use_container_width=True,
            on_click=lambda: state.chosen_estimators.clear(),
        )


def show_params_values():
    st.write(state.params_values)


def show_estimators():
    # st.subheader("Models")
    st.dataframe(
        pd.DataFrame(state.chosen_estimators, index=["estimator", "params"]).T,
        use_container_width=True,
    )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    init("chosen_estimators", dict())
    init("params_values", dict())
    init("_estimator", None)

    if "chosen_estimators" not in state:
        state["chosen_estimators"] = dict()

    st.header("Models")
    show_estimators()
    show_add_model()
    show_params_inputs()
