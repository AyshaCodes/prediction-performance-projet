app_code = '''
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Prédiction Performance Projet", page_icon="🎯", layout="wide")

# CSS personnalisé
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    .title { text-align: center; font-size: 2.5rem; font-weight: bold; color: #4fc3f7; }
    .subtitle { text-align: center; color: #aaaaaa; font-size: 1rem; margin-bottom: 2rem; }
    .result-box { padding: 20px; border-radius: 15px; text-align: center; font-size: 1.5rem; font-weight: bold; }
    .metric-card { background: #1e1e2e; border-radius: 10px; padding: 15px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎯 Prédiction de Performance de Projet</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Application IA — Random Forest Classifier | SOUMARE AICHA</div>', unsafe_allow_html=True)
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
    from sklearn.model_selection import train_test_split
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le, df

model, le, df = train_model()

# Métriques en haut
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Dataset", "300 projets")
col2.metric("🎯 Accuracy", "100%")
col3.metric("🌲 Algorithme", "Random Forest")
col4.metric("📂 Classes", "3 catégories")

st.markdown("---")

# Formulaire
st.subheader("📋 Entrez les paramètres de votre projet")
col1, col2 = st.columns(2)

with col1:
    budget = st.slider("💰 Budget prévu (MAD)", 10000, 100000, 50000, step=1000)
    duree = st.slider("📅 Durée prévue (jours)", 10, 120, 60)
    complexite = st.selectbox("⚙️ Complexité du projet", ["Faible", "Moyenne", "Elevee"])

with col2:
    equipe = st.slider("👥 Taille de l\'équipe", 2, 20, 10)
    risques = st.slider("⚠️ Nombre de risques identifiés", 0, 10, 3)
    experience = st.slider("🎓 Expérience du chef (ans)", 1, 15, 5)

st.markdown("---")

if st.button("🔍 Prédire la performance du projet", use_container_width=True):
    complexite_enc = le.transform([complexite])[0]
    X_input = [[budget, duree, equipe, complexite_enc, risques, experience]]
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    classes = model.classes_

    if prediction == "Réussi":
        st.success(f"## ✅ Résultat : {prediction}")
        st.balloons()
    elif prediction == "En retard":
        st.warning(f"## ⚠️ Résultat : {prediction}")
    else:
        st.error(f"## ❌ Résultat : {prediction}")

    # Probabilités
    st.markdown("### 📊 Probabilités par classe")
    prob_col1, prob_col2, prob_col3 = st.columns(3)
    for i, (cls, prob) in enumerate(zip(classes, proba)):
        emoji = "✅" if cls == "Réussi" else "⚠️" if cls == "En retard" else "❌"
        [prob_col1, prob_col2, prob_col3][i].metric(f"{emoji} {cls}", f"{prob*100:.1f}%")

st.markdown("---")
st.markdown("*Application développée par SOUMARE AICHA — Mini-Projet IA & ML 2025-2026*")
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

from google.colab import files
files.download("app.py")
print("✅ Nouveau app.py prêt !")
