# Dual-Branch Guided Adaptive Feature Enhancement Network for Semantic Segmentation of Remote Sensing Images

## 1. Introduction
LAGNet is an open-source semantic segmentation toolbox built on PyTorch, PyTorch Lightning, and timm, specifically dedicated to developing advanced deep learning models for remote sensing image semantic segmentation.

## 2. Dataset Introduction
This project mainly targets high-resolution remote sensing image semantic segmentation tasks, conducting model training, validation, and evaluation on standard benchmark datasets including ISPRS Vaihingen, ISPRS Potsdam, and LoveDA.

* **ISPRS Vaihingen Dataset**: Consists of high-resolution aerial imagery along with corresponding Digital Surface Models (DSM). It covers common land-cover categories such as buildings, trees, low vegetation, cars, and impervious surfaces, serving primarily to evaluate model performance on fine-grained boundaries and complex spatial distributions.
* **ISPRS Potsdam Dataset**: Contains large-format, high-resolution top-down aerial images (offering RGB and Near-Infrared/NIR channels). With extensive spatial coverage and rich surface details, it is widely used to evaluate multi-scale feature extraction and semantic understanding in complex high-resolution remote sensing scenarios.
* **LoveDA Dataset**: A challenging land-cover dataset containing high-resolution (0.3 m) imagery gathered from both urban and rural areas across three cities (Nanjing, Changzhou, and Wuhan). It encompasses 7 primary land-cover categories: building, road, water, barren, forest, agriculture, and background. Featuring distinct domain shifts between urban and rural scenes, it is widely utilized for evaluating model robustness, multi-scale feature representation, and generalizability in diverse environments.

## 3. Environment Installation
Create and configure the Python environment using Linux Terminal:

```bash
# Create and activate conda environment
conda create -n airs python=3.8
conda activate airs

# Install PyTorch with CUDA 11.8 support
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install project dependencies
pip install -r GeoSeg/requirements.txt
