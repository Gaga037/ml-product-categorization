import pandas as pd
import os
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

#Lokacija trenutnog fajla (data/)
current_dir = os.path.dirname(__file__)

#Ucitaj dataset
csv_path = os.path.join(current_dir, "products.csv")
df = pd.read_csv(csv_path)

#Ocisti nazive kolona
df.columns = df.columns.str.strip()

#Zadrzi samo relevantne kolone i izbaci prazne vrednosti
df = df[['Product Title', 'Category Label']].dropna()

#Ciscenje teksta (lowercase, bez brojeva, spec. znakova)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # ukloni specijalne znakove
    text = re.sub(r'\d+', '', text)      # ukloni brojeve
    return text

df['clean_title'] = df['Product Title'].apply(clean_text)

#Enkodovanje ciljne promenljive
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['Category Label'])

#TF-IDF vektorizacija
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X = tfidf.fit_transform(df['clean_title'])
y = df['category_encoded']

#Podela na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Treniraj model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

#Sacuvaj model, vektorizer i label encoder u ../model/
model_dir = os.path.join(current_dir, "../model")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "model.pkl"))
joblib.dump(tfidf, os.path.join(model_dir, "vectorizer.pkl"))
joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

print(" Model uspesno treniran i sacuvan u folder 'model/'")

