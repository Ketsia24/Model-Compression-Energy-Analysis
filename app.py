# app.py - Dashboard TP3 Green AI (adapté à Gra_ket_darryl.xlsx)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Green AI Dashboard", page_icon="🌱", layout="wide")
st.title("🌱 Dashboard Green AI — TP3 Pruning & Quantization")

# ----------------- FONCTIONS UTILES -----------------
def convert_numeric(col):
    """Convertit proprement une série en numérique (gère texte, virgules, datetime)."""
    if pd.api.types.is_datetime64_any_dtype(col):
        # Si Excel a enregistré un temps comme datetime, on convertit en millisecondes
        return (
            col.dt.hour * 3600000 +
            col.dt.minute * 60000 +
            col.dt.second * 1000 +
            col.dt.microsecond / 1000.0
        )
    col = col.astype(str).str.replace(",", ".").str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(col, errors="coerce")

def normalize(series, invert=False):
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return pd.Series(0, index=s.index)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0, index=s.index)
    norm = (s - mn) / (mx - mn)
    return 1 - norm if invert else norm

def pareto_mask(df, acc="Top1_Acc", co2="Emissions_kgCO2eq"):
    """Retourne un masque booléen des points Pareto-optimaux."""
    data = df[[acc, co2]].dropna()
    mask = pd.Series(False, index=df.index)
    if data.empty:
        return mask
    data = data.sort_values([acc, co2], ascending=[False, True])
    best_co2 = np.inf
    pareto_idx = []
    for idx, row in data.iterrows():
        if row[co2] <= best_co2:
            pareto_idx.append(idx)
            best_co2 = row[co2]
    mask.loc[pareto_idx] = True
    return mask

# ----------------- CHARGEMENT DU FICHIER -----------------
uploaded = st.file_uploader("📤 Importer ton fichier Excel (Gra_ket_darryl.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Dépose ton fichier Excel pour démarrer.")
    st.stop()

df = pd.read_excel(uploaded)

# Renommage cohérent
rename_map = {
    "Énergie (kWh)": "Energy_kWh",
    "Taille (MB)ModelSize_MB": "ModelSize_MB",
    "Temps (s)": "TrainTime_s",
    "Observations clés": "Notes"
}
df = df.rename(columns=rename_map)

# Conversion des colonnes numériques
num_cols = ["Top1_Acc", "Energy_kWh", "Emissions_kgCO2eq", "ModelSize_MB", "TrainTime_s", "Latency_ms_per_image"]
for c in num_cols:
    if c in df.columns:
        df[c] = convert_numeric(df[c])

st.subheader("Aperçu du jeu de données")
st.dataframe(df, use_container_width=True)

# ----------------- FILTRES -----------------
st.sidebar.header("🎛️ Filtres")
backbones = sorted(df["Backbone"].dropna().unique().tolist())
scenarios = sorted(df["Scenario"].dropna().unique().tolist())
bb_sel = st.sidebar.multiselect("Backbone", backbones, default=backbones)
sc_sel = st.sidebar.multiselect("Scenario", scenarios, default=scenarios)

flt = df.copy()
if bb_sel:
    flt = flt[flt["Backbone"].isin(bb_sel)]
if sc_sel:
    flt = flt[flt["Scenario"].isin(sc_sel)]

# ----------------- SCORE GLOBAL -----------------
st.sidebar.header("⚖️ Pondération du Score global")
w_acc = st.sidebar.slider("Poids Accuracy (↑)", 0.0, 1.0, 0.5, 0.05)
w_co2 = st.sidebar.slider("Poids CO₂ (↓)", 0.0, 1.0, 0.3, 0.05)
w_lat = st.sidebar.slider("Poids Latence (↓)", 0.0, 1.0, 0.1, 0.05)
w_size = st.sidebar.slider("Poids Taille (↓)", 0.0, 1.0, 0.1, 0.05)

flt["Score_Global"] = (
    w_acc * normalize(flt["Top1_Acc"]) -
    w_co2 * normalize(flt["Emissions_kgCO2eq"]) -
    w_lat * normalize(flt["Latency_ms_per_image"]) -
    w_size * normalize(flt["ModelSize_MB"])
)

# ----------------- METRIQUES -----------------
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🔝 Accuracy max", f"{flt['Top1_Acc'].max():.2%}")
c2.metric("🌍 CO₂ min", f"{flt['Emissions_kgCO2eq'].min():.6f} kg")
c3.metric("⚡ Énergie totale", f"{flt['Energy_kWh'].sum():.5f} kWh")
c4.metric("⏱️ Temps total", f"{flt['TrainTime_s'].sum():.0f} s")
c5.metric("📦 Taille médiane", f"{flt['ModelSize_MB'].median():.2f} MB")

# ----------------- VISUALISATIONS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Accuracy vs CO₂ (Pareto)",
    "📦 Taille vs Accuracy",
    "⏳ Latence vs Énergie",
    "🏁 Décision finale"
])

with tab1:
    st.subheader("📈 Frontière de Pareto (Accuracy vs CO₂)")
    mask = pareto_mask(flt)
    flt["Pareto"] = np.where(mask, "Pareto-front", "Autres")
    fig = px.scatter(
        flt, x="Emissions_kgCO2eq", y="Top1_Acc",
        color="Backbone", symbol="Pareto", size="ModelSize_MB",
        hover_data=["Scenario", "Energy_kWh", "Latency_ms_per_image", "Notes"],
        labels={"Emissions_kgCO2eq": "CO₂eq (kg)", "Top1_Acc": "Accuracy (%)"}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📦 Taille du modèle vs Accuracy")
    fig2 = px.scatter(
        flt, x="ModelSize_MB", y="Top1_Acc", color="Backbone",
        hover_data=["Scenario", "Emissions_kgCO2eq", "Notes"],
        labels={"ModelSize_MB": "Taille (MB)", "Top1_Acc": "Accuracy"}
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("⏳ Latence vs Énergie")
    fig3 = px.scatter(
        flt, x="Energy_kWh", y="Latency_ms_per_image", color="Backbone",
        size="Top1_Acc", hover_data=["Scenario", "Emissions_kgCO2eq"],
        labels={"Energy_kWh": "Énergie (kWh)", "Latency_ms_per_image": "Latence (ms/image)"}
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("🏁 Classement des modèles")
    df_sorted = flt.sort_values("Score_Global", ascending=False)
    st.dataframe(df_sorted, use_container_width=True)
    st.download_button(
        "💾 Télécharger le classement (CSV)",
        data=df_sorted.to_csv(index=False).encode("utf-8"),
        file_name="TP3_dashboard_ranking.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("📘 Ce dashboard est optimisé pour ton fichier Excel Gra_ket_darryl(2).xlsx — toutes les colonnes sont reconnues automatiquement.")
