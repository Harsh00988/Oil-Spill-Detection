FROM python:3.8.10-slim

WORKDIR /app

# Install Linux package dependencies
RUN apt-get update && apt-get install -y wget

# Create the necessary directories
RUN mkdir -p /data/models /data/images

# Download model file from Google Drive using the file ID
RUN wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1pQbUxKs5MQ5HgUrONPsytkEOUmMOlRRn' -O /data/models/oil_spill_seg_resnet_50_deeplab_v3+_80.pt

# Copy sample image and requirements file
COPY ./sample_padding_image_for_inference/img_0814.jpg /data/images/img_0814.jpg
COPY ./requirements_deployment.txt .

# Install Python dependencies
RUN pip install -r requirements_deployment.txt

# Copy training and app files
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

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080"]
