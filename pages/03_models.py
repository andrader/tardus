import numbers
from numbers import Integral

import pandas as pd
import sklearn.utils._param_validation as pv
import streamlit as st
from sklearn.utils import all_estimators
from streamlit import session_state as state
import helpers as h
from tardus.parameters import _isinstance_of
from tardus.parameters import get_params_constraints


@st.cache_data
def get_all_regressors_sklearn(type_filter=None):
    return dict(all_estimators(type_filter=type_filter))


def input_selectbox(c, default, options=None, **kwargs):
    if options is None:
        options = list(c.options)
    widget = st.selectbox
    widget_kwargs = dict(
        options=options,
        index=options.index(default) if default in options else 0,
    )
    return widget, widget_kwargs


def input_multiselect(c, default, **kwargs):
    pass


def input_number(c, default, **kwargs):
    convert = int if c.type is Integral else float
    invalids = (None, float("-inf"), float("inf"))

    min_value = None if c.left in invalids else convert(c.left)
    max_value = None if c.right in invalids else convert(c.right)
    value = convert(0) if default in invalids else convert(default)

    if (min_value is not None) and (value < min_value):
        value = min_value

    if (max_value is not None) and (value > max_value):
        value = max_value

    widget = st.number_input
    widget_kwargs = dict(
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=1e-1 if convert is float else 1,
        format="%.6f" if convert is float else "%d",
        # help=str(c),
        # label="value",
        # label_visibility="visible",
    )
    return widget, widget_kwargs


def input_text(c, default, **kwargs):
    widget = st.text_input
    widget_kwargs = dict(value=default, help=str(c))
    return widget, widget_kwargs


def input_toogle(c, default, **kwargs):
    widget = st.toggle
    widget_kwargs = dict(
        # options = [False, True],
        value=default,
        help=str(c),
        label_visibility="visible",
    )
    return widget, widget_kwargs


def dipatch_contraint(c, default, **kwargs):
    """Dispatch the constraint into the appropriate create widget function."""

    if _isinstance_of(c, (numbers.Integral, numbers.Real)) and not _isinstance_of(
        c, bool
    ):
        type = numbers.Integral if _isinstance_of(c, numbers.Integral) else numbers.Real
        c = pv.Interval(type, None, None, closed="neither")

    if isinstance(c, pv._NoneConstraint) or c is pv._NoneConstraint:
        return lambda **x: None, dict()
    if isinstance(c, pv._Booleans) or _isinstance_of(c, bool):
        return input_toogle(c, default, **kwargs)
    if isinstance(c, pv.Interval):
        return input_number(c, default, **kwargs)
    if isinstance(c, pv.Options):
        return input_selectbox(c, default, **kwargs)
    if _isinstance_of(c, str):
        return input_text(c, "" if default is None else default, **kwargs)
    if _isinstance_of(c, dict):
        widget = st.data_editor
        widget_kwargs = dict(
            drop_label=None,
            data=dict() if default is None else default,
            use_container_width=True,
        )
        return widget, widget_kwargs

    st.write(f"Unsuported: `{str(c)}`")
    return lambda **kwargs: default, dict()


def create_params_widgets(estimator_name, params):
    params_values = {}

    containers = h.grid(ncols=3)
    for param in params.items():
        p_name, (p_constraints, p_default, p_ignored_constraints) = param
        key_usedefault = "_".join([estimator_name, p_name, "usedefault"])
        key_value = "_".join([estimator_name, p_name, "value"])
        key_constraint = " ".join([estimator_name, p_name, "constraint"])

        if p_default == "deprecated":
            st.write(f"`{p_name}` ignored: deprecated")
            continue

        if p_ignored_constraints:
            st.write(
                f"`{p_name}` ignored:",
                ", ".join(
                    [f"{x} (`{type(x).__name__}`)" for x in p_ignored_constraints]
                ),
            )

        container = next(containers)
        with container:
            with st.expander(f"**{p_name.replace('_', ' ').title()}**", expanded=True):
                if len(p_constraints) == 0:
                    c = pv._NoneConstraint
                else:
                    # first c defaults to c that initial value satyisfies
                    idx = [c.is_satisfied_by(p_default) for c in p_constraints]
                    idx = idx.index(True) if True in idx else 0
                    # st.write(p_name, p_default, new_default, idx)

                    c_idx = st.selectbox(
                        label=key_constraint,
                        options=list(range(len(p_constraints))),
                        index=idx,
                        key=key_constraint,
                        format_func=lambda i: str(p_constraints[i]),
                        label_visibility="hidden",
                        disabled=len(p_constraints) == 1,
                    )
                    c = p_constraints[c_idx]

                try:
                    res = dipatch_contraint(c, p_default)
                    widget, widget_kwargs = res
                except Exception as e:
                    # st.write(res)
                    # st.write(f"{p_name=}, {str(c)=}, {c=}, {p_default=}, {p_constraints=}")
                    raise e

                _kwargs = dict(label=str(p_name), label_visibility="hidden")
                _kwargs.update(widget_kwargs)
                if "drop_label" in widget_kwargs:
                    del _kwargs["label"]
                    del _kwargs["label_visibility"]
                    del _kwargs["drop_label"]

                val = widget(
                    key=key_value,
                    on_change=callback_update_use_default,
                    args=(p_default, key_usedefault, key_value),
                    # so label_visibility  can be overwritted
                    **_kwargs,
                )

                use_default = st.checkbox(
                    f"Use default `{p_default}`",
                    value=True,
                    key=key_usedefault,
                    help=f"Ignore input value and use the default `{p_default}`",
                )

                params_values[p_name] = p_default if use_default else val

    # overwrite defaults with user input
    defaults = {k: v[1] for k, v in params.items()}
    params_values = dict(defaults, **params_values)

    return params_values


def callback_add_estimator():
    _estimator_name = state._estimator_name
    _estimator_cls = state._estimator_cls
    params = state.params_values

    try:
        estimator = _estimator_cls().set_params(**params)
    except Exception:
        # st.write(str(type(_estimator_cls)), params)
        # st.error(str(e))
        estimator = _estimator_cls(**params)

    state.chosen_estimators[state.estimator_label] = dict(
        estimator=estimator,
        _params=params,
        _cls=_estimator_cls,
    )
    st.toast("Added")


def callback_update_use_default(default, key_usedefault, key_value):
    state[key_usedefault] = False  # (value is default) or (value == default)


def show_params_inputs():
    _estimator_name = state._estimator_name
    _estimator_cls = state._estimator_cls

    params = get_params_constraints(_estimator_cls)

    select_params = st.multiselect(
        "Select parameters",
        options=list(params.keys()),
        default=list(params.keys()),
        help="Select model parameters",
    )

    state.params_values = create_params_widgets(
        estimator_name=_estimator_name,
        params={k: v for k, v in params.items() if k in select_params},
    )


def show_add_model():
    st.subheader("Add model")

    cols = st.columns(2)

    with cols[0]:
        type_filter = st.multiselect(
            "Estimator type",
            options={"classifier", "regressor", "cluster", "transformer"},
            key="select_type_filter",
            format_func=str.title,
        )
        estimators = get_all_regressors_sklearn(type_filter if type_filter else None)

        def update_selected_estimator():
            state._estimator_cls = estimators[state._estimator_name]

        estimator_name = st.selectbox(
            "Select model class",
            options=[None] + list(estimators),
            key="_estimator_name",
            on_change=update_selected_estimator,
        )
        if not estimator_name:
            st.stop()

    with cols[1]:
        st.text_input("Label", estimator_name, key="estimator_label")

        cols = st.columns(2)
        with cols[0]:
            st.button(
                "Overwrite"
                if state.estimator_label in state.chosen_estimators
                else "Add",
                use_container_width=True,
                on_click=callback_add_estimator,
                disabled=not state.estimator_label.strip(),
            )

        with cols[1]:
            st.button(
                "Delete",
                use_container_width=True,
                disabled=state.estimator_label not in state.chosen_estimators,
                on_click=lambda: state.chosen_estimators.pop(
                    state.estimator_label, None
                ),
            )


def show_estimators():
    cols = st.columns(4)
    with cols[0]:
        st.subheader("Models")

    with cols[-1]:
        st.button(
            "Delete ALL models",
            use_container_width=True,
            on_click=lambda: state.chosen_estimators.clear(),
        )

    chosen_estimators = {
        k: {kk: str(vv) for kk, vv in v.items()}
        for k, v in state.chosen_estimators.items()
    }

    st.dataframe(
        pd.DataFrame(chosen_estimators).T.reset_index(names="name"),
        hide_index=True,
        column_order=["name", "estimator"],
        # pd.DataFrame(chosen_estimators, index=[1,2]).T,#, index=["estimator", "params"]).T,
        use_container_width=True,
    )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    h.init("chosen_estimators", dict())
    h.init("params_values", dict())
    h.init("select_type_filter", [])
    h.init("_estimator_cls", None)
    h.init("_estimator_name", None)

    st.header("Models")
    show_estimators()
    show_add_model()
    show_params_inputs()

    # Create pipeline
