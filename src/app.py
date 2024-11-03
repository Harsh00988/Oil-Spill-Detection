import os
import io
import numpy as np
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from PIL import Image
from skimage.io import imread
import torch.nn.functional as F
from training.metrics import *
from training.seg_models import *
from training.image_preprocessing import ImagePadder
from training.logger_utils import load_dict_from_json
from training.dataset import get_dataloader_for_inference
import rasterio
from rasterio.windows import Window
import folium
from streamlit_folium import folium_static

def run_inference(
    image_array,
    file_weights,
    num_classes=5,
    file_stats_json="training/image_stats.json",
):
    """
    ---------
    Arguments
    ---------
    image_array : ndarray
        a numpy array of the image
    file_weights : str
        full path to weights file
    num_classes : int
        number of classes in the dataset
    file_stats_json : str
        full path to the json stats file for preprocessing

    -------
    Returns
    -------
    pred_mask_arr : ndarray
        a numpy array of the prediction mask
    """
    # Initialize model and device
    oil_spill_seg_model = ResNet50DeepLabV3Plus(
        num_classes=num_classes, pretrained=True
    )
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oil_spill_seg_model.to(device)
    oil_spill_seg_model.load_state_dict(torch.load(file_weights, map_location=device))
    oil_spill_seg_model.eval()

    # Map for only the oil_spill and oil_spill_look_alike classes, with the same color
    dict_label_to_color_mapping = {
        1: np.array([0, 255, 255]),  # Color for oil_spill
        2: np.array([0, 255, 255]),  # Color for oil_spill_look_alike
    }

    # Load preprocessing stats
    try:
        dict_stats = load_dict_from_json(file_stats_json)
    except:
        dir_json = os.path.dirname(os.path.realpath(__file__))
        dict_stats = load_dict_from_json(os.path.join(dir_json, file_stats_json))

    # Initialize padding class
    try:
        image_padder = ImagePadder("/data/images")
    except:
        image_padder = ImagePadder("./sample_padding_image_for_inference")

    # Debug print to check image dimensions
    print(f"Original image shape: {image_array.shape}")

    # Apply padding and preprocessing
    image_padded = image_padder.pad_image(image_array)
    print(f"Padded image shape: {image_padded.shape}")

    # Normalize image
    image_preprocessed = image_padded / 255.0
    image_preprocessed = image_preprocessed - dict_stats["mean"]
    image_preprocessed = image_preprocessed / dict_stats["std"]
    image_preprocessed = np.expand_dims(image_preprocessed, axis=0)
    image_preprocessed = np.transpose(image_preprocessed, (0, 3, 1, 2))

    # Convert image to tensor and run inference
    image_tensor = torch.tensor(image_preprocessed).float()
    image_tensor = image_tensor.to(device, dtype=torch.float)
    pred_logits = oil_spill_seg_model(image_tensor)
    pred_probs = F.softmax(pred_logits, dim=1)
    pred_label = torch.argmax(pred_probs, dim=1)

    # Convert the prediction to numpy and initialize mask
    pred_label_arr = pred_label.detach().cpu().numpy().squeeze()
    pred_mask_arr = np.zeros((pred_label_arr.shape[0], pred_label_arr.shape[1], 3))

    # Fill mask only for oil_spill and oil_spill_look_alike classes
    for sem_class in [1, 2]:  # Only process oil_spill and oil_spill_look_alike
        curr_class_label = (pred_label_arr == sem_class).astype(np.uint8)
        curr_class_label = curr_class_label[:, :, np.newaxis]

        # Map the color for the selected classes
        curr_class_color_mapping = dict_label_to_color_mapping[sem_class]
        curr_class_color_mapping = curr_class_color_mapping.reshape(1, 3)

        # Apply color to the mask
        pred_mask_arr += curr_class_label * curr_class_color_mapping

    # Convert to uint8 and remove padding
    pred_mask_arr = pred_mask_arr.astype(np.uint8)
    padded_height, padded_width = pred_label_arr.shape
    pred_mask_arr = pred_mask_arr[11 : padded_height - 11, 15 : padded_width - 15]

    return pred_mask_arr

def show_mask_interpretation():
    colors = ["#00FFFF"]
    labels = ["oil_spill"]
    my_cmap = ListedColormap(colors, name="my_cmap")
    data = [[1, 2, 3, 4, 5]]
    fig = plt.figure(figsize=(20, 2))
    plt.title("Oil Spill mask interpretation")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels)
    plt.yticks([])
    plt.imshow(data, cmap=my_cmap)
    st.pyplot(fig)
    return

def infer():
    st.title("Oil spill detection app")

    file_weights_default = "/data/models/oil_spill_seg_resnet_50_deeplab_v3+_80.pt"
    file_weights = st.sidebar.text_input("File model weights", file_weights_default)

    if not os.path.isfile(file_weights):
        st.write("Wrong weights file path")
    else:
        st.write(f"Weights file: {file_weights}")

    # select an input SAR image file
    image_file_buffer = st.sidebar.file_uploader(
        "Select input SAR image", type=["jpg", "jpeg", "tif", "tiff"]
    )
    # read the image
    if image_file_buffer is not None:
        if image_file_buffer.name.endswith((".tif", ".tiff")):
            with rasterio.open(image_file_buffer) as src:
                image_array = src.read()
                image_array = np.transpose(image_array, (1, 2, 0))
                if src.crs:
                    st.write("Georeferenced TIFF detected. Displaying on basemap.")
                    m = folium.Map(location=[src.bounds.top, src.bounds.left], zoom_start=10)
                    folium.raster_layers.ImageOverlay(
                        image=image_array,
                        bounds=[[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]],
                        opacity=0.6
                    ).add_to(m)
                    folium_static(m)
        else:
            image = Image.open(image_file_buffer)
            image = image.resize((1250, 650))  # Resize to (650, 1250)
            image_array = np.array(image)
        st.image(image_array, caption=f"Input image: {image_file_buffer.name}")
    else:
        st.write("Input image: not selected")

    # select a mask image file
    mask_file_buffer = st.sidebar.file_uploader(
        "Select groundtruth mask image (optional, only for visual comparison with the prediction)",
        type=["png"],
    )
    # read the mask
    if mask_file_buffer is not None:
        mask = Image.open(mask_file_buffer)
        mask_array = np.array(mask)
        st.image(mask_array, caption=f"Mask image: {mask_file_buffer.name}")
    else:
        st.write("Groundtruth mask image (optional): not selected")

    # run inference when the option is invoked by the user
    infer_button = st.sidebar.button("Run inference")
    if infer_button:
        if image_file_buffer.name.endswith((".tif", ".tiff")):
            with rasterio.open(image_file_buffer) as src:
                height, width, _ = image_array.shape
                window_size = (1250, 650)
                stride = 625
                mask_pred_array = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(0, height - window_size[1], stride):
                    for x in range(0, width - window_size[0], stride):
                        window = Window(x, y, window_size[0], window_size[1])
                        window_image = src.read(window=window)
                        window_image = np.transpose(window_image, (1, 2, 0))
                        mask_pred = run_inference(window_image, file_weights)
                        mask_pred_array[y:y+window_size[1], x:x+window_size[0]] = mask_pred
                mask_pred_image = Image.fromarray(mask_pred_array.astype("uint8"), "RGB")
        else:
            mask_pred = run_inference(image_array, file_weights)
            mask_pred_image = Image.fromarray(mask_pred.astype("uint8"), "RGB")

        st.image(
            mask_pred_image,
            caption=f"Predicted mask for the input: {image_file_buffer.name}",
        )

        # option to download predicted mask
        with io.BytesIO() as file_obj:
            mask_pred_image.save(file_obj, format="PNG")
            mask_for_download = file_obj.getvalue()
        st.download_button(
            "Download predicted mask",
            data=mask_for_download,
            file_name="pred_mask.png",
            mime="image/png",
        )

        # display a figure showing the interpretation of the mask labels
        show_mask_interpretation()
    return

def app_info():
    st.title("App info")
    st.markdown("Task - Oil Spill segmentation")
    st.write("This app uses a ResNet50 DeepLabV3+ model for segmentation and visualizes predictions.")
    return

app_modes = {
    "App Info": app_info,
    "Oil Spill Inference App": infer,
}

def start_app():
    selected_mode = st.sidebar.selectbox("Select mode", list(app_modes.keys()))
    app_modes[selected_mode]()
    return

def main():
    start_app()
    return

if __name__ == "__main__":
    main()