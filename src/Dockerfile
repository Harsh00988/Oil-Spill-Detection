FROM python:3.8.10-slim

WORKDIR /app

# Install linux package dependencies

# Install google-cloud-storage package
RUN pip install google-cloud-storage

# Copy only necessary files for the build process
COPY ./requirements_deployment.txt .

# Install Python dependencies
RUN pip install -r requirements_deployment.txt

# Copy the rest of the application code
COPY ./training/__init__.py ./training/
COPY ./training/dataset.py ./training/
COPY ./training/encoder_models.py ./training/
COPY ./training/decoder_models.py ./training/
COPY ./training/image_preprocessing.py ./training/
COPY ./training/logger_utils.py ./training/
COPY ./training/metrics.py ./training/
COPY ./training/seg_models.py ./training/
COPY ./training/image_stats.json ./training/
COPY ./app.py .

# Download the model from GCS
RUN mkdir -p /data/models
RUN python -c "from google.cloud import storage; storage.Client().bucket('oil-spill-bucket').blob('oil_spill_seg_resnet_50_deeplab_v3+_80.pt').download_to_filename('/data/models/oil_spill_seg_resnet_50_deeplab_v3+_80.pt')"

# Copy sample padding image
COPY ./sample_padding_image_for_inference/img_0814.jpg /data/images/img_0814.jpg

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080"]
