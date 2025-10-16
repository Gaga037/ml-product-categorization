import joblib
import os
import re

#Lokacija fajlova
current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, "../model")

#Ucitaj sacuvane fajlove
model = joblib.load(os.path.join(model_dir, "model.pkl"))
tfidf = joblib.load(os.path.join(model_dir, "vectorizer.pkl"))
le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

#Funkcija za ciscenje teksta
def clean(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# Unos proizvoda
title = input("Unesi naziv proizvoda: ")
cleaned = clean(title)
X = tfidf.transform([cleaned])

#Predikcija
pred = model.predict(X)
category = le.inverse_transform(pred)[0]

print(f"Predvidjena kategorija: {category}")
