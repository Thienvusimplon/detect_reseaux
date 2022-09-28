import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
import os
import numpy as np
import streamlit as st

st.title("Detect réseau")

def load_image(image_file):
    img = Image.open(image_file)
    return img

image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg", "JPG"])
model = torch.hub.load('Thienvusimplon/detect_reseaux', 'custom', path='best.pt', force_reload=True, skip_validation=True ) 
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
number = 1

if image_file != None:
    predictions = model(load_image(image_file))
    st.write(predictions.pandas().xyxy[0])

    for i in range(len(predictions.crop())):
        label_confidence = predictions.crop()[i]["label"].split()       
        st.subheader(f"Crop {number}")
        st.write("Label détecté :", label_confidence[0])
        st.write("Score de confidence : ", label_confidence[1])
        img = Image.fromarray(predictions.crop()[i]["im"])
        st.image(img)

        img = img_transform(img).unsqueeze(0)

        logits = parseq(img)

        pred = logits.softmax(-1)
        label, confidence = parseq.tokenizer.decode(pred)
        st.write('Decoded label = {}'.format(label[0]))
        st.write('Decoded confidence = {}'.format(confidence[0]))
        number +=1
    st.image(Image.fromarray(predictions.render()[0]))