# Digit classification model

## Overview

Classify digits.

## Folder structure

```
├── checkpoints
   └── model.pt
├── data
   └── test_data
├── data_loader
   └── data_loaders.py
├── models
   └── LeNet5.py
├── notebook
   └── MNIST.ipynb
├── requirements
   └── requirements.yaml
├── utils
   └── utils.py
├── README.md
├── app.py
├── test.py
└── train.py
```

## Clone and install requirements

```
git clone https://github.com/yongwookim1/catvsdog_image_classification.git
conda env create -f requirements/requirements.yaml
```

## Test

```
python test.py
```

## Train

```
python train.py [-h] [--seed SEED] [--epochs EPOCHS] [--lr LR] [--batch_size BATCH_SIZE]
```

## Demo

```
streamlit run app.py --server.fileWatcherType none --server.port [YOUR_PORT]
```
