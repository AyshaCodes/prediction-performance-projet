import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="PredictProject · IA",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%);
}

/* Hero */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #38bdf8);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shine 3s linear infinite;
    margin-bottom: 0.3rem;
}
@keyframes shine {
    to { background-position: 200% center; }
}
.hero-sub {
    text-align: center;
    color: #64748b;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2rem;
}

/* Stat cards */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, border-color 0.2s;
}
.stat-card:hover {
    transform: translateY(-3px);
    border-color: rgba(56,189,248,0.4);
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #38bdf8;
}
.stat-lbl {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* Section titles */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(56,189,248,0.2);
}

/* Result cards */
.result-success {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
    border: 1px solid rgba(34,197,94,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-warning {
    background: linear-gradient(135deg, rgba(234,179,8,0.15), rgba(234,179,8,0.05));
    border: 1px solid rgba(234,179,8,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-danger {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin-bottom: 0.5rem;
}
.result-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
}

/* Slider labels */
.field-label {
    font-size: 0.8rem;
    color: #94a3b8;
    font-weight: 500;
    margin-bottom: 0.2rem;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), transparent);
    margin: 1.5rem 0;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    color: #0a0e1a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(56,189,248,0.3) !important;
}

/* Prob bars */
.prob-bar-wrap {
    margin: 0.4rem 0;
}
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #94a3b8;
    margin-bottom: 0.2rem;
}
.prob-bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}

/* Footer */
.footer {
    text-align: center;
    color: #334155;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}

/* Hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Entraînement du modèle ───────────────────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    data = []
    for _ in range(100):
        data.append({
            "budget_prevu":    np.random.randint(10000, 100000),
            "duree_prevue":    np.random.randint(10, 120),
            "taille_equipe":   np.random.randint(8, 20),
            "complexite":      np.random.choice(["Faible", "Moyenne"]),
            "nb_risques":      np.random.randint(0, 2),
            "experience_chef": np.random.randint(8, 15),
            "performance":     "Réussi"
        })
    for _ in range(100):
        data.append({
            "budget_prevu":    np.random.randint(10000, 100000),
            "duree_prevue":    np.random.randint(10, 120),
            "taille_equipe":   np.random.randint(5, 15),
            "complexite":      np.random.choice(["Moyenne", "Elevee"]),
            "nb_risques":      np.random.randint(3, 6),
            "experience_chef": np.random.randint(4, 8),
            "performance":     "En retard"
        })
    for _ in range(100):
        data.append({
            "budget_prevu":    np.random.randint(10000, 100000),
            "duree_prevue":    np.random.randint(10, 120),
            "taille_equipe":   np.random.randint(2, 8),
            "complexite":      "Elevee",
            "nb_risques":      np.random.randint(7, 10),
            "experience_chef": np.random.randint(1, 4),
            "performance":     "Échoué"
        })

    df = pd.DataFrame(data)
    le = LabelEncoder()
    df["complexite_enc"] = le.fit_transform(df["complexite"])
    features = ["budget_prevu","duree_prevue","taille_equipe",
                "complexite_enc","nb_risques","experience_chef"]
    X = df[features]
    y = df["performance"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le, df, features, round(acc*100, 1)

model, le, df, features, accuracy = train_model()

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎯 PredictProject</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Application IA de prédiction de performance · SOUMARE AICHA</div>',
            unsafe_allow_html=True)

# ── STATS ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-val">300</div>
    <div class="stat-lbl">Projets analysés</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{accuracy}%</div>
    <div class="stat-lbl">Précision</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">RF</div>
    <div class="stat-lbl">Random Forest</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">3</div>
    <div class="stat-lbl">Classes prédites</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── FORMULAIRE ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Paramètres du projet</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="field-label">💰 Budget prévu (MAD)</div>', unsafe_allow_html=True)
    budget = st.slider("", 10000, 100000, 50000, step=1000, key="budget", label_visibility="collapsed")

    st.markdown('<div class="field-label">📅 Durée prévue (jours)</div>', unsafe_allow_html=True)
    duree = st.slider("", 10, 120, 60, key="duree", label_visibility="collapsed")

with col2:
    st.markdown('<div class="field-label">👥 Taille de l\'équipe</div>', unsafe_allow_html=True)
    equipe = st.slider("", 2, 20, 10, key="equipe", label_visibility="collapsed")

    st.markdown('<div class="field-label">⚠️ Nombre de risques</div>', unsafe_allow_html=True)
    risques = st.slider("", 0, 10, 3, key="risques", label_visibility="collapsed")

with col3:
    st.markdown('<div class="field-label">🎓 Expérience du chef (ans)</div>', unsafe_allow_html=True)
    experience = st.slider("", 1, 15, 5, key="experience", label_visibility="collapsed")

    st.markdown('<div class="field-label">⚙️ Complexité du projet</div>', unsafe_allow_html=True)
    complexite = st.selectbox("", ["Faible", "Moyenne", "Elevee"], key="complexite",
                               label_visibility="collapsed")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── BOUTON PRÉDIRE ────────────────────────────────────────────────────────────
col_btn = st.columns([1, 2, 1])[1]
with col_btn:
    predict = st.button("⚡ Prédire la performance", use_container_width=True)

# ── RÉSULTAT ──────────────────────────────────────────────────────────────────
if predict:
    complexite_enc = le.transform([complexite])[0]
    X_input = pd.DataFrame(
        [[budget, duree, equipe, complexite_enc, risques, experience]],
        columns=features
    )
    prediction = model.predict(X_input)[0]
    proba      = model.predict_proba(X_input)[0]
    classes    = list(model.classes_)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Résultat de la prédiction</div>', unsafe_allow_html=True)

    col_res, col_prob = st.columns([1, 1])

    with col_res:
        if prediction == "Réussi":
            css_class, emoji, color = "result-success", "✅", "#22c55e"
            st.balloons()
        elif prediction == "En retard":
            css_class, emoji, color = "result-warning", "⚠️", "#eab308"
        else:
            css_class, emoji, color = "result-danger", "❌", "#ef4444"

        st.markdown(f"""
        <div class="{css_class}">
            <div class="result-label">Performance prédite</div>
            <div class="result-value" style="color:{color}">{emoji} {prediction}</div>
            <div style="font-size:0.85rem;color:#64748b;margin-top:0.8rem;">
                Basé sur {len(df)} projets analysés
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_prob:
        st.markdown("**Probabilités par classe**")
        colors_map = {"Réussi": "#22c55e", "En retard": "#eab308", "Échoué": "#ef4444"}
        for cls, prob in sorted(zip(classes, proba), key=lambda x: -x[1]):
            c = colors_map.get(cls, "#38bdf8")
            pct = round(prob * 100, 1)
            st.markdown(f"""
            <div class="prob-bar-wrap">
                <div class="prob-label"><span>{cls}</span><span>{pct}%</span></div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct}%;background:{c}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── GRAPHIQUES ────────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Analyse des données</div>', unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0d1526")
        ax.set_facecolor("#0d1526")
        counts = df["performance"].value_counts()
        colors_pie = ["#22c55e", "#eab308", "#ef4444"]
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index,
            autopct="%1.0f%%", colors=colors_pie,
            textprops={"color": "#94a3b8", "fontsize": 9},
            wedgeprops={"edgecolor": "#0d1526", "linewidth": 2}
        )
        for at in autotexts:
            at.set_color("#e2e8f0")
        ax.set_title("Répartition des performances", color="#e2e8f0", fontsize=11, pad=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_g2:
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0d1526")
        ax.set_facecolor("#0d1526")
        imp = pd.Series(model.feature_importances_, index=features).nlargest(6).sort_values()
        bars = ax.barh(imp.index, imp.values, color="#38bdf8", edgecolor="none", height=0.6)
        ax.set_title("Importance des variables", color="#e2e8f0", fontsize=11)
        ax.tick_params(colors="#64748b", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d45")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    PredictProject · SOUMARE AICHA · Mini-Projet IA & ML 2024-2025 · Random Forest · Scikit-learn
</div>
""", unsafe_allow_html=True)
