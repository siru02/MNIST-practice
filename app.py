import streamlit as st
import io
import numpy as np
from PIL import Image


import torch
from torchvision.transforms import v2

st.set_page_config(layout="wide")


def transform_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("L")
    image = np.array(image)
    image = torch.tensor(image)
    image = image.view(-1, 1, 28, 28)

    test_transforms = v2.Compose(
        [
            v2.Resize(32, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.1307], std=[0.3081]),
        ]
    )
    transformed_img = test_transforms(image)
    return transformed_img


def get_prediction(model, image_bytes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformed_image = transform_image(image_bytes=image_bytes).to(device)
    outputs = model(transformed_image)
    y_hat = torch.argmax(outputs, 1)
    return transformed_image, y_hat


def main():
    st.title("Digit Classification Model")

    model = torch.load("checkpoints/model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    uploaded_file = st.file_uploader("Upload digit image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        st.image(image, caption="Uploaded Image")

        _, y_hat = get_prediction(model, image_bytes)
        label = y_hat[0]

        st.header(f"The output number is {label}")


if __name__ == "__main__":
    main()
