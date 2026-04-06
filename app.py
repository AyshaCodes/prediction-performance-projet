app_code = '''
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prédiction Performance Projet", page_icon="🎯")
st.title("🎯 Prédiction de Performance de Projet")
st.markdown("---")

@st.cache_resource
def train_model():
    np.random.seed(42)
    data = []
    for _ in range(100):
        b = np.random.randint(10000, 100000)
        d = np.random.randint(10, 120)
        data.append({"budget_prevu": b, "duree_prevue": d,
            "taille_equipe": np.random.randint(8, 20),
            "complexite": np.random.choice(["Faible", "Moyenne"]),
            "nb_risques": np.random.randint(0, 2),
            "experience_chef": np.random.randint(8, 15),
            "performance": "Réussi"})
    for _ in range(100):
        b = np.random.randint(10000, 100000)
        d = np.random.randint(10, 120)
        data.append({"budget_prevu": b, "duree_prevue": d,
            "taille_equipe": np.random.randint(5, 15),
            "complexite": np.random.choice(["Moyenne", "Elevee"]),
            "nb_risques": np.random.randint(3, 6),
            "experience_chef": np.random.randint(4, 8),
            "performance": "En retard"})
    for _ in range(100):
        b = np.random.randint(10000, 100000)
        d = np.random.randint(10, 120)
        data.append({"budget_prevu": b, "duree_prevue": d,
            "taille_equipe": np.random.randint(2, 8),
            "complexite": "Elevee",
            "nb_risques": np.random.randint(7, 10),
            "experience_chef": np.random.randint(1, 4),
            "performance": "Échoué"})
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df["complexite_enc"] = le.fit_transform(df["complexite"])
    X = df[["budget_prevu", "duree_prevue", "taille_equipe", "complexite_enc", "nb_risques", "experience_chef"]]
    y = df["performance"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le

model, le = train_model()

st.subheader("📋 Paramètres du projet")
col1, col2 = st.columns(2)

with col1:
    budget = st.slider("💰 Budget prévu (MAD)", 10000, 100000, 50000, step=1000)
    duree = st.slider("📅 Durée prévue (jours)", 10, 120, 60)
    complexite = st.selectbox("⚙️ Complexité", ["Faible", "Moyenne", "Elevee"])

with col2:
    equipe = st.slider("👥 Taille de l\'équipe", 2, 20, 10)
    risques = st.slider("⚠️ Nombre de risques", 0, 10, 3)
    experience = st.slider("🎓 Expérience chef (ans)", 1, 15, 5)

st.markdown("---")

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
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

from google.colab import files
files.download("app.py")
files.download("dataset_performance_projets.csv")
print("✅ Fichiers prêts !")
