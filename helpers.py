import streamlit as st


def init(k, v, d=st.session_state):
    if k not in d:
        d[k] = v


def grid(ncols: int = 2, n: int = None):
    parent = st.container()
    i = 0
    is_infinite = n is None
    while is_infinite or (i < n):
        col = i % ncols
        if col == 0:
            cols = parent.columns(ncols)
        yield cols[col]
        i += 1
