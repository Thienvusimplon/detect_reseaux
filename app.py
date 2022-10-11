import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
import os
import numpy as np
import streamlit as st
import shutil
import cv2
import pandas as pd

st.title("Detect réseaux")

def load_image(image_file):
    img = Image.open(image_file)
    return img


model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True, skip_validation=True ) 
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg", "JPG"])

crop_number = 1

if image_file != None:
    predictions = model(load_image(image_file))
    st.write(predictions.pandas().xyxy[0])

    df = pd.DataFrame(predictions.pandas().xyxy[0])
    index = len(predictions.crop()) - 1
    df = df.rename(columns={"confidence": "confiance couleur", "name": "couleur"})
    df = df.drop([df.columns[5]], axis=1)
    df["x_center"] = (df["xmax"] + df["xmin"]) / 2
    df["y_center"] = (df["ymax"] + df["ymin"]) / 2

    for i in range(len(predictions.crop())):     
        label_confidence = predictions.crop()[i]["label"].split() 
        st.subheader(f"Crop {crop_number}")
        st.write("Couleur détecté :", label_confidence[0])
        st.write("Score de confiance couleur: ", label_confidence[1]) 

        afficher_crop_img = cv2.cvtColor(predictions.crop()[i]["im"], cv2.COLOR_BGR2RGB)
        st.image(afficher_crop_img)

        detect_number_img = Image.fromarray(predictions.crop()[i]["im"])
        detect_number_img_2 = img_transform(detect_number_img).unsqueeze(0)
        logits = parseq(detect_number_img_2)
        pred = logits.softmax(-1)
        label, confidence = parseq.tokenizer.decode(pred)
        st.write('Détection chiffre = {}'.format(label[0]))
        st.write('Score de confiance du chiffre détecté = {}'.format(confidence)) 
          
        df.at[index, "numero"] = label[0]
        #Confidence du numéro à extraire
        #st.write(type(confidence[0]))
        #df.at[i, "confidence numero"] = confidence[0]

        index -= 1
        crop_number += 1
    
    df = df[["xmin", "xmax", "ymin", "ymax", "x_center", "y_center", "confiance couleur", "couleur", "numero"]]
    st.image(Image.fromarray(predictions.render()[0]))
    st.write(df)
    shutil.rmtree('runs/')

    csv = df.to_csv().encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f"{image_file.name}.csv",
        mime='text/csv',
    )
