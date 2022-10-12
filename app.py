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

# Chargement des modèles
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True, skip_validation=True ) 
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

# Upload l'image à traiter
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg", "JPG"])

# Traitement de l'image lorsqu'elle est chargée
if image_file != None:

    # Recherche des centres d'intérêt + detection de couleur
    predictions = model(Image.open(image_file))

    # Preparation du csv pour download
    df = pd.DataFrame(predictions.pandas().xyxy[0])
    df = df.rename(columns={"confidence": "confiance couleur", "name": "couleur"})
    df = df.drop([df.columns[5]], axis=1)
    df["x_center"] = (df["xmax"] + df["xmin"]) / 2
    df["y_center"] = (df["ymax"] + df["ymin"]) / 2

    # Affichage des centres d'intérêt détecté
    index = len(predictions.crop()) - 1
    crop_number = 1
    for i in range(len(predictions.crop())):     
        label_confidence = predictions.crop()[i]["label"].split() 
        st.subheader(f"Crop {crop_number}")
        st.write("Couleur détecté :", label_confidence[0])
        st.write("Score de confiance couleur: ", label_confidence[1]) 
        afficher_crop_img = cv2.cvtColor(predictions.crop()[i]["im"], cv2.COLOR_BGR2RGB)
        st.image(afficher_crop_img)

        # Détection des numéros à partir de nos centres d'intérêt
        detect_number_img = Image.fromarray(predictions.crop()[i]["im"])
        detect_number_img_2 = img_transform(detect_number_img).unsqueeze(0)
        logits = parseq(detect_number_img_2)
        pred = logits.softmax(-1)
        label, confidence = parseq.tokenizer.decode(pred)
        st.write('Détection chiffre = {}'.format(label[0]))
        st.write('Score de confiance du chiffre détecté = {}'.format(np.mean(confidence[0].detach().numpy())))

        # Ajout des colonnes numero et confidence numero pour le csv à download
        df.at[index, "numero"] = label[0]
        df.at[index, "confidence numero"] = np.mean(confidence[0].detach().numpy())

        index -= 1
        crop_number += 1
    
    # Réarangement de l'ordre des colonnes du csv
    df = df[["xmin", "xmax", "ymin", "ymax", "x_center", "y_center", "couleur", "confiance couleur", "numero", "confidence numero"]]

    # Affichage de la photo avec les boundings box
    st.image(Image.fromarray(predictions.render()[0]))

    # Affichage du dataframe
    st.write(df)

    # Effacer le dossier runs
    shutil.rmtree('runs/')

    # Préparation du download du csv
    csv = df.to_csv().encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f"{image_file.name}.csv",
        mime='text/csv',
    )
