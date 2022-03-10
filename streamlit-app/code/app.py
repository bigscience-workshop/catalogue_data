import os
from functools import partial
import streamlit as st
import json
import diff_viewer
import pandas as pd
from collections import defaultdict, OrderedDict
from datasets import load_from_disk
import pprint as pp

DATASET_DIR_PATH_BEFORE_CLEAN_SELECT = "/home/lucile/data"
CLEANING_VERSIONS = [
    "clean_v2",
    "clean_v1"
]
st.set_page_config(page_title="Dataset explorer", layout="wide")

st.write("The purpose of this application is to sequentially view the changes made to a dataset.")



def get_ds(ds_path):
    ds = load_from_disk(ds_path)
    return ds

def on_click_next():
    st.session_state['idx'] += 1
    st.session_state['idx'] = st.session_state['idx'] % len(st.session_state['ds'])

def on_click_previous():
    st.session_state['idx'] -= 1
    st.session_state['idx'] = st.session_state['idx'] % len(st.session_state['ds'])

def on_ds_change(ds_path):
    st.session_state['ds'] =  get_ds(ds_path)
    st.session_state['idx'] = 0
    st.session_state['ds_name'] =  ds_path

def get_logs_stats(log_path):
    try:
        data = OrderedDict({
            "Order": [],
            "Name": [],
            "Initial number of samples" : [],
            "Final number of samples" : [],
            "Initial size in bytes" : [],
            "Final size in bytes" : [],
        })
        with open(log_path) as f:
            content = f.read()
        # subcontent = [line for line in content.split("\n") if "main" in line]
        # st.write(subcontent)
        metric_dict = defaultdict(lambda: {})
        order = 0
        for line in content.split("\n"):
            for metric_name in list(data.keys()) + ["Applied filter", "Applied deduplication function", "Applied map function"]:

                if metric_name == "Name" or metric_name == "Order":
                    continue

                if metric_name not in line:
                    continue

                if metric_name == "Removed percentage" and "Removed percentage in bytes" in line:
                    continue

                if metric_name == "Deduplicated percentage" and "Deduplicated percentage in bytes" in line:
                    continue

                value = line.split(metric_name)[1].split(" ")[1]

                if metric_name in ["Applied filter", "Applied deduplication function", "Applied map function"]:
                    operation_name = value
                    metric_dict[operation_name]["Order"] = order
                    order += 1
                    continue

                assert metric_name not in metric_dict[operation_name], f"operation_name: {operation_name}\n\nvalue: {value}\n\nmetric_dict: {pp.pformat(metric_dict)} \n\nmetric_name: {metric_name} \n\nline: {line}"
                metric_dict[operation_name][metric_name] = value
        for name, data_dict in metric_dict.items():
            for metric_name in data.keys():
                if metric_name == "Name":
                    data[metric_name].append(name)
                    continue

                data[metric_name].append(data_dict[metric_name])
        df = pd.DataFrame(data)
        df.rename({"Initial size in bytes": "Initial size (GB)", "Final size in bytes": "Final size (GB)"}, axis=1, inplace=True)
        df["% samples removed"] = (df["Initial number of samples"].astype(float) - df["Final number of samples"].astype(float)) /  df["Initial number of samples"].astype(float) *100
        df["Size (GB) % removed"] = (df["Initial size (GB)"].astype(float) - df["Final size (GB)"].astype(float)) /  df["Initial size (GB)"].astype(float) *100
        st.dataframe(df)
    except Exception as e:
        st.write(e)
        st.write("Subset of the logs:")
        subcontent = [line for line in content.split("\n") if "INFO - __main__" in line and "Examples of" not in line and "Examples n°" not in line]
        st.write(subcontent)


col_option_clean, col_option_ds = st.columns(2)

with col_option_clean:
    option_clean = st.selectbox(
        'Select the cleaning version',
        CLEANING_VERSIONS,
    )
    DATASET_DIR_PATH = os.path.join(DATASET_DIR_PATH_BEFORE_CLEAN_SELECT, option_clean)

dataset_names = sorted(list(os.listdir(DATASET_DIR_PATH)))
with col_option_ds:
    option_ds = st.selectbox(
        'Select the dataset',
        dataset_names,
    )

checks_path = os.path.join(DATASET_DIR_PATH, option_ds, "checks")
checks_names = sorted(list(os.listdir(checks_path)))

log_path = os.path.join(DATASET_DIR_PATH, option_ds, "logs.txt")
get_logs_stats(log_path=log_path)

option_check = st.selectbox(
     'Select the operation applied to inspect',
     checks_names,
)
ds_path = os.path.join(checks_path,option_check)

# Initialization
if 'ds' not in st.session_state:
    st.session_state['ds'] =  get_ds(ds_path)
    st.session_state['ds_name'] =  ds_path
    st.session_state['idx'] = 0

if ds_path != st.session_state['ds_name']:
    on_ds_change(ds_path)

if len(st.session_state['ds']) == 1000:
    st.warning("Attention only a subset of size 1000 of the modified / filtered examples can be shown in this application")
with st.expander("See details of the available checks"):
    st.write(st.session_state['ds'])


if "_filter_" in option_check:
    idx_1 = st.session_state['idx']
    idx_2 = (st.session_state['idx'] + 1) % len(st.session_state['ds'])
    text_1 = st.session_state['ds'][idx_1]['text']
    text_2 = st.session_state['ds'][idx_2]['text']


    st.markdown(f"<h1 style='text-align: center'>Some examples of filtered out texts</h1>", unsafe_allow_html=True)
    col_button_previous , _, col_button_next = st.columns(3)

    col_button_next.button("Go to next example", key=None, help=None, on_click=on_click_next, args=None, kwargs=None)
    col_button_previous.button("Go to previous example", key=None, help=None, on_click=on_click_previous, args=None, kwargs=None)
    col_1, col_2 = st.columns(2)
    with col_1:
        st.subheader(f"Example n°{idx_1}")
        if 'meta' in st.session_state['ds'][st.session_state['idx']]:
            with st.expander("See meta field of the example"):
                meta = st.session_state['ds'][st.session_state['idx']]['meta']
                st.write(meta)
        text_1_show = text_1.replace("\n", "<br>")
        st.markdown(f'<div>{text_1_show}</div>', unsafe_allow_html=True)

    with col_2:
        st.subheader(f"Example n°{idx_2}")
        if 'meta' in st.session_state['ds'][st.session_state['idx']]:
            with st.expander("See meta field of the example"):
                meta = st.session_state['ds'][st.session_state['idx']+1]['meta']
                st.write(meta)
        text_2_show = text_2.replace("\n", "<br>")
        st.markdown(f'<div>{text_2_show}</div>', unsafe_allow_html=True)

else:
    col_button_previous, col_title, col_button_next = st.columns(3)
    col_title.markdown(f"<h1 style='text-align: center'>Example n°{st.session_state['idx']}</h1>", unsafe_allow_html=True)
    col_button_next.button("Go to next example", key=None, help=None, on_click=on_click_next, args=None, kwargs=None)
    col_button_previous.button("Go to previous example", key=None, help=None, on_click=on_click_previous, args=None, kwargs=None)

    text = st.session_state['ds'][st.session_state['idx']]['text']
    old_text = st.session_state['ds'][st.session_state['idx']]['old_text']
    st.markdown(f"<h2 style='text-align: center'>Changes applied</h1>", unsafe_allow_html=True)
    col_text_1, col_text_2 = st.columns(2)
    with col_text_1:
        st.subheader("Old text")
    with col_text_2:
        st.subheader("New text")
    diff_viewer.diff_viewer(old_text=old_text, new_text=text, lang="none")
    if 'meta' in st.session_state['ds'][st.session_state['idx']]:
        with st.expander("See meta field of the example"):
            meta = st.session_state['ds'][st.session_state['idx']]['meta']
            st.write(meta)

    with st.expander("See full old and new texts of the example"):
        col_1, col_2 = st.columns(2)
        with col_1:
            st.subheader("Old text")
            old_text_show = old_text.replace("\n", "<br>")
            st.markdown(f'<div>{old_text_show}</div>', unsafe_allow_html=True)
        with col_2:
            st.subheader("New text")
            text_show = text.replace("\n", "<br>")
            st.markdown(f'<div>{text_show}</div>', unsafe_allow_html=True)
