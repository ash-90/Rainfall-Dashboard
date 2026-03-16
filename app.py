import os
import random

import kagglehub
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image

#page config 
st.set_page_config(
    page_title="World Rainfall Dashboard",
    page_icon="☔️",
    layout="wide",
)

#data loading helpers 
@st.cache_resource(show_spinner="Downloading dataset from Kaggle…")
def load_dataset_path() -> str:
    return kagglehub.dataset_download("darsh22blc1378/world-rainfall-dataset")


@st.cache_data(show_spinner=False)
def get_image_paths(data_path: str) -> list[str]:
    img_dir = os.path.join(data_path, "imagesTrain")
    files = sorted(
        [f for f in os.listdir(img_dir) if f.endswith(".png")],
        key=lambda x: int(x.removesuffix(".png")),
    )
    return [os.path.join(img_dir, f) for f in files]


@st.cache_data(show_spinner=False)
def load_image_array(path: str) -> np.ndarray:
    return np.array(Image.open(path))


@st.cache_data(show_spinner="Computing mean rainfall map…")
def compute_mean_image(paths: tuple, sample_size: int = 2000) -> np.ndarray:
    sampled = random.sample(list(paths), min(sample_size, len(paths)))
    arrays = [load_image_array(p) for p in sampled]
    return np.mean(arrays, axis=0).astype(np.uint8)


#load data 
data_path = load_dataset_path()
image_paths = get_image_paths(data_path)
world_bmp_path = os.path.join(data_path, "world.bmp")

#header 
st.title("Global Rainfall Distribution")
st.markdown("""
Data source: [NASA World Rainfall Dataset](https://www.kaggle.com/datasets/darsh22blc1378/world-rainfall-dataset)

Visualised with Streamlit and Plotly

""")

st.subheader("Mean Rainfall Intensity Map")
st.markdown(
    "Average pixel values computed from a random sample of **2000 images** "
    "to approximate the mean global rainfall distribution."
)

mean_arr = compute_mean_image(tuple(image_paths))

cmap = st.selectbox(
    "Colormap",
    ["turbo", "viridis", "Blues", "plasma", "RdYlBu_r"],
    key="mean_cmap",
)

#correct the red channel 
red_channel = mean_arr[:, :, 0]
red_channel_corrected = 255 - red_channel

fig_mean = px.imshow(
    red_channel_corrected,
    color_continuous_scale=cmap,
    labels={"color": "Mean intensity"},
    title="Mean Rainfall Intensity (n=2000)",
    aspect="equal",
)
fig_mean.update_layout(
    height=520,
    margin=dict(l=0, r=0, t=40, b=0),
    coloraxis_colorbar=dict(
        title="Mean intensity",
        thickness=12,
        len=0.82,
    ),
)
st.plotly_chart(fig_mean, use_container_width=True)



# composite graph
fig_composite = px.imshow(
    mean_arr,
    title="Mean RGB Composite (n=2000)",
    aspect="equal",
)
fig_composite.update_layout(
    height=520,
    margin=dict(l=0, r=0, t=40, b=0),
    coloraxis_showscale=False,
)
st.plotly_chart(fig_composite, use_container_width=True)

st.divider()
st.caption(
    "COR2252 Weather and Climate, Group 2"
)