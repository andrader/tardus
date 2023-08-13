from typing import Any
import streamlit as st
from streamlit import session_state as state
import pandas as pd
import numpy as np
import operator as op

# operator expects input type
# -no value
# -single value:
#   - text: selectbox
#   - number: slider + input
# -2 values:
#   - number: slider, ...
# -mult values:
#   - text: multiselect

OPS = {
    # (op, inputnumber, )
    "In": (lambda x, y: x.isin(y), -1),
    "Contains": lambda x, y: x.str.contains(y, case=False),
    "Equals": op.eq,
    "Is_NA": (pd.isna, 0),
    "Between": (lambda x, y: x.between(y[0], y[1], inclusive="both"), 2),
}

OPS_STR = ["In", "Contains", "Equals", "Is_NA"]
OPS_NUM = ["Equals", "Between", "Is_NA"]


@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://github.com/datablist/sample-csv-files/raw/main/files/customers/customers-10000.csv"
    )
    return dict(df=df)


def create_key(*args, sep="_"):
    return sep.join([str(a) for a in args])


def select_operators(dtype, key_op):
    if pd.api.types.is_string_dtype(dtype):
        ops = OPS_STR
    elif pd.api.types.is_numeric_dtype(dtype):
        ops = OPS_NUM
    else:
        ops = OPS

    op_name = st.selectbox("Operator", ops, key=key_op, label_visibility="hidden")

    operator = OPS[op_name]
    operator = operator if isinstance(operator, tuple) else (operator, 1)
    return op_name, *operator


class Filter:
    def __init__(self, column_name) -> None:
        self.column_name = column_name
        self.key_op = create_key("filter", column_name, "op")
        self.key_val = create_key("filter", column_name, "val")

    def __call__(self, df: pd.DataFrame) -> Any:
        pass

    def create_widget(self, df: pd.DataFrame):
        col_name = self.column_name
        col: pd.Series = df[col_name]
        dtype = col.dtype
        nunique = col.nunique()
        key_op = self.key_op
        key_val = self.key_val

        with st.expander(col_name):
            op_name, operator, nargs, *_ = select_operators(dtype, key_op)

            widget_kwargs = dict()
            if nargs == 0:
                widget = lambda *args, **kwargs: None
            elif nargs == 1:
                if pd.api.types.is_numeric_dtype(dtype):
                    min_value, max_value = col.min(), col.max()
                    widget = st.number_input
                    widget_kwargs = dict(
                        min_value=min_value,
                        max_value=max_value,
                    )
                elif pd.api.types.is_string_dtype(dtype):
                    widget = st.text_input
                else:
                    raise NotImplementedError
            elif nargs == 2:
                min_value, max_value = col.min(), col.max()
                widget = st.slider
                widget_kwargs = dict(
                    min_value=min_value,
                    max_value=max_value,
                    value=(min_value, max_value),
                )
            elif nargs == -1:
                widget = st.multiselect
                widget_kwargs = dict(options=col.unique())
            else:
                widget = st.text_input
                widget_kwargs = dict()

            widget("Values", **widget_kwargs, key=key_val, label_visibility="hidden")

    def get_mask(self, df):
        key_op = self.key_op
        key_val = self.key_val
        col = df[self.column_name]

        if key_val in state and key_op in state:
            val = state[key_val]

            operator = OPS[state[key_op]]
            operator = operator if isinstance(operator, tuple) else (operator, 1)
            operator, nargs, *_ = operator

            return operator(col, val)

        return np.ones_like(col, bool)


def update_filters(columns):
    state.filters = {c: Filter(c) for c in columns}


def draw_filters(df: pd.DataFrame) -> pd.DataFrame:
    columns = st.multiselect("Columns to filter", df.columns)

    update_filters(columns)

    import numpy as np

    _f: Filter
    msks = [np.ones((df.shape[0],), bool)]
    accum_msks = [msks[0]]
    filtered_df = df
    for _f in state.filters.values():
        _f.create_widget(filtered_df)

        msk = _f.get_mask(df)
        msks.append(msk)

        accum_msk = op.and_(accum_msks[-1], msk)
        accum_msks.append(accum_msk)

        filtered_df = df[accum_msk]

    return filtered_df


############## APP

st.set_page_config(
    layout="wide",
)

if "data" not in state or st.button("Load data"):
    state["data"] = load_data()

if "filters" not in state:
    state["filters"] = dict()

select_dataset = st.selectbox("Dataset", [None] + list(state["data"].keys()))
if select_dataset:
    df = state.data[select_dataset]

    # draw filters
    filtered_df = draw_filters(df)

    st.dataframe(filtered_df)
