import streamlit as st
import pandas as pd
import sklearn.datasets
import io
import requests
from operator import methodcaller
import helpers as h
import numpy as np

state = st.session_state

TOY_DATASETS = [
    "breast_cancer",
    "diabetes",
    "digits",
    "iris",
    "linnerud",
    "wine",
]


@st.cache_data
def load_examples():
    get_loader = lambda name: methodcaller(
        "load_" + name, return_X_y=False, as_frame=True
    )(sklearn.datasets)
    sklearn.datasets.load_breast_cancer()
    datasets = {name: get_loader(name) for name in TOY_DATASETS}
    return datasets


def request_content(url):
    response = requests.get(url)
    response.raise_for_status()
    return io.BytesIO(response.content)


def read_from_file_or_url(file, **kwargs):
    file_name = file.name
    if file_name.endswith(".csv"):
        reader = pd.read_csv
    elif file_name.endswith(".xlsx"):
        reader = pd.read_excel
    elif file_name.endswith(".parquet"):
        reader = pd.read_parquet
    else:
        if isinstance(file, str):
            content = request_content(file)
            return read_from_file_or_url(content, **kwargs)

        st.error("Unsupported file format")
        return

    return reader(file, **kwargs)


def callback_load_data():
    data = dict()
    if state.upload_from == "file":
        file = state.file_uploader
        file_name = state.file_upload_name
        if file and file_name:
            df = read_from_file_or_url(file)
            data = {file_name: df}
    elif state.upload_from == "examples":
        data = load_examples()
    else:
        raise ValueError("invalid upload_from type")

    if data:
        state.datasets.update(data)
    else:
        st.write("No data to load")


def callback_update_name():
    if state.file_uploader is not None:
        state.file_upload_name = (
            state.file_upload_name
            if state.file_upload_name == ""
            else state.file_uploader.name
        )


# Main Streamlit app
def show_load_data():
    st.title("Dataset Loader")

    from_options = ["examples", "file"]
    upload_from = st.selectbox(
        "Select from", from_options, key="upload_from", format_func=str.title
    )

    if upload_from == "file":
        st.file_uploader(
            "Choose a file/url",
            type=["csv", "xlsx", "parquet", "application/zip"],
            key="file_uploader",
            on_change=callback_update_name,
        )

        st.text_input(
            "Name of the dataset",
            key="file_upload_name",
        )

    elif upload_from == "examples":
        st.multiselect("Examples", TOY_DATASETS, TOY_DATASETS, disabled=True)

    st.button(
        "Load",
        on_click=callback_load_data,
        # disabled=(state.file_uploader is None) or not state.file_upload_name,
    )


def show_dict_of_dfs(d, level=3, format_func=str.title, expander=False):
    # tabs = st.tabs(d)

    get_container = (
        lambda k: st.expander(k, expanded=False) if expander else st.container()
    )

    for k, v in d.items():
        cont = get_container(k)
        with cont:
            st.write(f"{level*'#'} {format_func(str(k))}")
            # with tab:
            if isinstance(v, dict):
                show_dict_of_dfs(v, level + 1)
            elif isinstance(v, pd.Series):
                st.dataframe(v)
            elif isinstance(v, pd.DataFrame):
                max_cols = 30
                max_rows = 100

                x = v
                if v.shape[1] > max_cols:
                    st.warning(f"Showing first {max_cols} cols")
                    x = x.iloc[:, :max_cols]

                if v.shape[0] > max_rows:
                    st.warning(f"Showing first {max_rows} rows")
                    x = x.iloc[:max_rows, :]

                st.dataframe(x, use_container_width=True)
            elif isinstance(v, (np.ndarray)):
                st.write(v)
            else:
                st.write(str(v))
    # return tabs


def show_added_datasets():
    cols = st.columns([8, 2])
    with cols[0]:
        st.subheader("Added Datasets")
    with cols[-1]:
        st.button(
            "Clear", on_click=lambda: state.datasets.clear(), use_container_width=True
        )
    show_dict_of_dfs(state.datasets or dict(), expander=True)


def show_select_ml_data():
    st.subheader("Select ML target and features")

    dataset_name = st.selectbox(
        "Dataset", [None] + list(state.datasets), key="selectbox_dataset_name"
    )

    dataset = state.datasets.get(dataset_name)

    if isinstance(dataset, dict):
        dataset = dataset.get("frame")

    if dataset is not None:
        cols = list(dataset.columns)
        idx_target = [i for i, c in enumerate(cols) if c.lower() == "target"]
        idx_target = idx_target[0] if idx_target else 0

        target_name = st.selectbox("Target", cols, idx_target)

        available_features = [c for c in cols if c != target_name]
        feature_names = st.multiselect(
            "Features",
            available_features,
            available_features,
        )

        data = {
            "dataset_name": dataset_name,
            "target_name": target_name,
            "feature_names": feature_names,
            "dataset": dataset,
            "features": dataset[feature_names],
            "target": dataset[target_name],
        }

        if st.button("Save"):
            state.data = data


@st.cache_resource
def show_summary(dataset_name):
    st.subheader("Summary - ML data")
    data = state.data
    if state.data:
        st.write(f'dataset_name: `{data["dataset_name"]}`')
        st.write(f'target_name: `{data["target_name"]}`')
        st.write(
            f"feature_names: " + " ".join([f"`{x}`" for x in data["feature_names"]])
        )
        # st.write(data["dataset"])
        st.write("features summary:")
        st.dataframe(data["features"].describe().T, use_container_width=True)
        st.write("target summary")
        st.dataframe(data["target"].describe().to_frame().T, use_container_width=True)

        # st.write()


if __name__ == "__main__":
    h.init("datasets", dict())
    h.init("data", dict())
    h.init("selectbox_dataset_name", None)

    show_load_data()
    st.divider()
    show_added_datasets()
    st.divider()
    show_select_ml_data()
    st.divider()
    show_summary(state.data.get("dataset_name"))
