Oil Spill Detection App
=======================

Overview
--------

This repository contains a Streamlit-based web application designed for detecting oil spills in Synthetic Aperture Radar (SAR) images using a deep learning model. The model is built on a ResNet50 backbone with a DeepLabV3+ architecture, specifically trained for semantic segmentation tasks. The application allows users to upload SAR images, run inference to detect oil spills, and visualize the results.

Features
--------

-   *Inference on SAR Images*: Upload SAR images (JPEG, TIFF) and run inference to detect oil spills.

-   *Georeferenced TIFF Support*: If the uploaded image is a georeferenced TIFF, it will be displayed on a basemap using Folium.

-   *Visual Comparison*: Optionally upload a ground truth mask for visual comparison with the predicted mask.

-   *Download Predicted Mask*: Download the predicted mask as a PNG file.

-   *Mask Interpretation*: Display a color legend explaining the mask labels.

Dataset
-------
-  *Dataset used for this model is SOS Dataset: Link to the dataset - https://drive.google.com/file/d/12grU_EAPbW75eyyHj-U5pOfnwQzm0MFw/view

Installation
------------

### Prerequisites

-   pip install -r requirements.txt

### Steps

1.  *Clone the Repository*


    git clone https://github.com/Harsh00988/Oil-Spill-Detection.git
    cd oil-spill-detection

2.  *Create a Virtual Environment* (Optional but recommended)


    python3 -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate

3.  *Install Dependencies*

    pip install -r requirements.txt

4.  *Download Model Weights*

    Place the model weights file (oil_spill_seg_resnet_50_deeplab_v3+_80.pt) in the /data/models/ directory.

Usage
-----

1.  *Run the Streamlit App*



    streamlit run app.py

2.  *Access the App*

    Open your web browser and go to http://localhost:8501 to access the application.

3.  *Upload SAR Image*

    -   Use the sidebar to upload a SAR image (JPEG, TIFF).

    -   Optionally, upload a ground truth mask for visual comparison.

4.  *Run Inference*

    -   Click the "Run inference" button to process the uploaded image.

    -   The predicted mask will be displayed, and you can download it as a PNG file.

5.  *View Mask Interpretation*

    -   A color legend will be displayed to explain the mask labels.

Directory Structure
-------------------


oil-spill-detection/
│
├── app.py                 # Main Streamlit application script
├── requirements.txt       # List of Python dependencies
├── README.md              # Project documentation (you are here)
├── data/
│   ├── models/
│   │   └── oil_spill_seg_resnet_50_deeplab_v3+_80.pt  # Model weights file
│   └── images/            # Directory for sample images (optional)
├── training/
│   ├── metrics.py         # Metrics calculation functions
│   ├── seg_models.py      # Model architecture definitions
│   ├── image_preprocessing.py  # Image preprocessing utilities
│   ├── logger_utils.py    # Utility functions for logging and loading JSON
│   ├── dataset.py         # Dataset handling functions
│   └── image_stats.json   # Image statistics for normalization
└── ...

Contributing
------------

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

Acknowledgments
---------------

-   The model architecture is based on ResNet50 and DeepLabV3+.

-   Special thanks to the Streamlit team for providing an easy-to-use framework for building data apps.