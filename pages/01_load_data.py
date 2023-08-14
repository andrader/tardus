import streamlit as st
import pandas as pd
import sklearn.datasets
import io
import requests
from operator import methodcaller
import helpers as h


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

        st.error("Unsupported state.file format")
        return

    return reader(file, **kwargs)


# Main Streamlit app
def main():
    state = st.session_state

    h.init("datasets", dict())
    h.init("dataset_name", "")
    h.init("_dataset", None)
    h.init("file", None)
    data_loader = None

    st.title("Dataset Loader")

    from_options = ["examples", "file"]
    upload_from = st.selectbox("Select from", from_options, format_func=str.title)

    if upload_from == "file":
        file_uploader = st.file_uploader(
            "Choose a file/url", type=["csv", "xlsx", "parquet", "application/zip"]
        )
        if file_uploader:
            state.file = file_uploader
            state.dataset_name = file_uploader.name
            data_loader = read_from_file_or_url

    elif upload_from == "examples":
        toy_datasets = [
            "breast_cancer",
            "diabetes",
            "digits",
            "iris",
            "linnerud",
            "wine",
        ]
        example_dataset = st.selectbox(
            "Choose an example dataset",
            toy_datasets,
            format_func=lambda x: x.replace("_", " ").title(),
        )
        state.file = example_dataset
        state.dataset_name = example_dataset
        get_loader = methodcaller(
            "load_" + state.dataset_name, return_X_y=False, as_frame=True
        )
        data_loader = lambda x: get_loader(sklearn.datasets)["frame"]

    state.dataset_name = st.text_input("Name of the dataset", value=state.dataset_name)

    if st.button("load dataset", disabled=state.file is None):
        state._dataset = data_loader(state.file)
        st.write(str(type(state._dataset)))

    if st.button(
        "add dataset",
        disabled=not (state.dataset_name and (state._dataset is not None)),
    ):
        st.write(state.dataset_name, str(type(state._dataset)))
        state.datasets[state.dataset_name] = state._dataset

    # Display added datasets
    st.header("Added Datasets")
    if state.datasets:
        tabs = st.tabs(state.datasets)
        for tab, (ds_name, ds) in zip(tabs, state.datasets.items()):
            with tab:
                st.write(ds_name)
                st.dataframe(ds.head(), use_container_width=True)

    st.button("Clear", on_click=lambda: state.datasets.clear())


if __name__ == "__main__":
    main()
