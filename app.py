
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Titre de l'application
st.set_page_config(page_title="Prédiction Performance Projet", page_icon="🎯")
st.title("🎯 Prédiction de Performance de Projet")
st.markdown("---")

# Génération et entraînement du modèle
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "budget_prevu": np.random.randint(10000, 100000, n),
        "duree_prevue": np.random.randint(10, 120, n),
        "taille_equipe": np.random.randint(2, 20, n),
        "complexite": np.random.choice(["Faible", "Moyenne", "Elevee"], n),
        "nb_risques": np.random.randint(0, 10, n),
        "experience_chef": np.random.randint(1, 15, n),
        "budget_reel": np.random.randint(10000, 120000, n),
        "duree_reelle": np.random.randint(10, 150, n),
    })
    def performance(row):
        if row["duree_reelle"] <= row["duree_prevue"] and row["budget_reel"] <= row["budget_prevu"]:
            return "Réussi"
        elif row["duree_reelle"] > row["duree_prevue"] * 1.3 or row["budget_reel"] > row["budget_prevu"] * 1.3:
            return "Échoué"
        else:
            return "En retard"
    df["performance"] = df.apply(performance, axis=1)
    le = LabelEncoder()
    df["complexite_enc"] = le.fit_transform(df["complexite"])
    X = df[["budget_prevu", "duree_prevue", "taille_equipe", "complexite_enc", "nb_risques", "experience_chef"]]
    y = df["performance"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le

model, le = train_model()

# Formulaire de saisie
st.subheader("📋 Paramètres du projet")
col1, col2 = st.columns(2)

with col1:
    budget = st.slider("💰 Budget prévu (MAD)", 10000, 100000, 50000, step=1000)
    duree = st.slider("📅 Durée prévue (jours)", 10, 120, 60)
    complexite = st.selectbox("⚙️ Complexité", ["Faible", "Moyenne", "Elevee"])

with col2:
    equipe = st.slider("👥 Taille de l'équipe", 2, 20, 10)
    risques = st.slider("⚠️ Nombre de risques", 0, 10, 3)
    experience = st.slider("🎓 Expérience chef (ans)", 1, 15, 5)

st.markdown("---")

# Prédiction
if st.button("🔍 Prédire la performance", use_container_width=True):
    complexite_enc = le.transform([complexite])[0]
    X_input = [[budget, duree, equipe, complexite_enc, risques, experience]]
    prediction = model.predict(X_input)[0]
    
    if prediction == "Réussi":
        st.success(f"## ✅ Projet : {prediction}")
        st.balloons()
    elif prediction == "En retard":
        st.warning(f"## ⚠️ Projet : {prediction}")
    else:
        st.error(f"## ❌ Projet : {prediction}")
