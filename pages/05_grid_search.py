"""
Ideia:
- usuário cria um pipeline
- adiciona steps ao pipeline
    - por exemplo scaler, preprocessing etc
- salva pipeline


validamos o pipeline se n

Com esse pipe
"""

import streamlit as st
import helpers as h

state = st.session_state


def add_to_list(key_val, key_vals):
    st.toast(f"Adding value {state[key_val]}")
    state[key_vals].append(state[key_val])


# PERSIST_KEY = "__PERSIST__"

# h.init(PERSIST_KEY, dict())
# pstate = state[PERSIST_KEY]

# def persist(key):
#     pstate[key] = state[key]


def update_list(key_key, value_key):
    state[key_key] = state[value_key]


def widget_parameter_space(parameter_name):
    def key(k):
        return "_".join(["", parameter_name, k])

    multis_key = key("multiselect")
    input_key = key("input")
    values_key = key("values")

    h.init(values_key, list())
    values = state[values_key]

    st.number_input(
        "Value",
        key=input_key,
        on_change=lambda: values.append(state[input_key])
        if state[input_key] not in values
        else None,
    )
    st.multiselect(
        "Parameter values",
        options=values,
        default=values,
        key=multis_key,
        on_change=update_list,
        kwargs=dict(key_key=values_key, value_key=multis_key),
    )

    return values


######### MAIN
h.init("parameter_space", dict())
pspace = state["parameter_space"]

if st.button("clear state"):
    state.clear()

param = "n_jobs"
with st.expander(f"**{param}**"):
    values = widget_parameter_space(param)
    if st.button("Save parameter space"):
        pspace[param] = values[:]

st.write(state)
