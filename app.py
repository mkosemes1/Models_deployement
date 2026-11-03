"""
Application Streamlit : 3 modèles de machine learning dans une seule interface.

Page 1 : Téléphones  -> état de l'annonce        (classification)
Page 2 : Voitures    -> Venant ou D'occasion      (classification)
Page 3 : Voitures    -> prix de vente             (régression)

Chaque page propose :
  - une prédiction unitaire (formulaire) ;
  - une prédiction sur un jeu de données complet (fichier CSV ou Excel).

Lancer l'application :  streamlit run app.py
Les fichiers .joblib produits par les notebooks doivent être dans le dossier  models/
"""
import warnings
from pathlib import Path

import joblib as jb
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

MODELS_DIR = Path(__file__).parent / "models"

CSS = """
<style>
    /* --- Palette et fond --- */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #1e293b;
    }
    /* --- Titres --- */
    h1, h2, h3 {
        font-family: 'Segoe UI', system-ui, sans-serif !important;
        color: #e2e8f0 !important;
        letter-spacing: -0.5px;
    }
    h1 { font-weight: 700 !important; }
    p, li, label, .stMarkdown {
        color: #cbd5e1 !important;
        font-family: 'Segoe UI', system-ui, sans-serif !important;
    }
    /* --- Carte du formulaire --- */
    [data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(8px);
    }
    /* --- Champs de saisie --- */
    [data-testid="stNumberInput"], [data-testid="stSelectbox"] {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
    }
    [data-baseweb="input"], [data-baseweb="select"] > div {
        background: #0f172a !important;
        border-color: #334155 !important;
        border-radius: 10px !important;
    }
    /* --- Boutons --- */
    .stButton > button, [data-testid="stFormSubmitButton"] > button,
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
        padding: 0.55rem 1.6rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(124, 58, 237, 0.45);
    }
    /* --- Métriques --- */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 700 !important; }
    /* --- Onglets --- */
    [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #334155; }
    [data-baseweb="tab"] {
        background: transparent; border-radius: 10px 10px 0 0;
        color: #94a3b8; font-weight: 600; padding: 8px 22px;
    }
    [aria-selected="true"] {
        background: rgba(37, 99, 235, 0.25) !important;
        color: #60a5fa !important;
        border-bottom: 2px solid #2563eb;
    }
    /* --- Messages --- */
    [data-testid="stSuccess"] {
        background: rgba(16, 185, 129, 0.15); border-left: 4px solid #10b981;
        border-radius: 10px; color: #a7f3d0;
    }
    [data-testid="stError"] {
        background: rgba(239, 68, 68, 0.15); border-left: 4px solid #ef4444;
        border-radius: 10px; color: #fecaca;
    }
    [data-testid="stWarning"] {
        background: rgba(245, 158, 11, 0.15); border-left: 4px solid #f59e0b;
        border-radius: 10px; color: #fde68a;
    }
    [data-testid="stInfo"] {
        background: rgba(59, 130, 246, 0.15); border-left: 4px solid #3b82f6;
        border-radius: 10px; color: #bfdbfe;
    }
    /* --- Dataframe --- */
    [data-testid="stDataFrame"] {
        border: 1px solid #334155; border-radius: 12px; overflow: hidden;
    }
    /* --- Uploader --- */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.7); border: 1px dashed #475569;
        border-radius: 12px; padding: 1rem;
    }
    /* --- Logo sidebar --- */
    .brand {
        text-align: center; padding: 1.2rem 0 0.4rem;
    }
    .brand .logo {
        font-size: 2.6rem; display: block; margin-bottom: 0.3rem;
    }
    .brand h2 {
        font-size: 1.15rem !important; font-weight: 700 !important;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 !important;
    }
    .brand p { font-size: 0.78rem; color: #64748b !important; margin-top: 2px !important; }
    .divider { border-top: 1px solid #1e293b; margin: 1rem 0; }
    /* --- Header page --- */
    .page-header {
        background: linear-gradient(90deg, rgba(37,99,235,0.25), rgba(124,58,237,0.12));
        border: 1px solid #334155; border-radius: 16px;
        padding: 1.3rem 1.8rem; margin-bottom: 1.4rem;
    }
    .page-header h1 { margin-bottom: 0.4rem !important; font-size: 1.9rem !important; }
    .page-header p { margin: 0 !important; color: #94a3b8 !important; font-size: 0.98rem; }
    /* --- Footer --- */
    .footer {
        text-align: center; color: #475569 !important; font-size: 0.8rem;
        padding: 2rem 0 1rem; border-top: 1px solid #1e293b; margin-top: 2.5rem;
    }
</style>
"""


def injecter_style():
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="logo">🤖</span>'
            "<h2>ML Predict Studio</h2>"
            "<p>Modèles de Machine Learning</p></div>"
            '<div class="divider"></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="footer">ML Predict Studio — Propulsé par Streamlit &amp; scikit-learn</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Pré-traitement propre à une page
# ----------------------------------------------------------------------
def nettoyer_adresse(df):
    """Même nettoyage que dans le notebook des voitures (colonne Adresse)."""
    df = df.copy()
    if "Adresse" in df.columns:
        df["Adresse"] = (
            df["Adresse"].astype("string")
            .str.replace(r"\s*,\s*", ",", regex=True)
            .str.strip()
            .replace({"Lac rose,Dakar": "Lac Rose,Dakar"})
        )
    return df


# ----------------------------------------------------------------------
# Configuration des 3 pages
#   prefix  : préfixe des fichiers .joblib de la page
#   numeric : plage et valeur par défaut des variables numériques
#             (plages = celles vues par le modèle à l'entraînement)
# ----------------------------------------------------------------------
PAGES = [
    {
        "key": "telephones",
        "prefix": "telephone",
        "notebook": "Clas_Log_Reg_Dec_Tree_Random_Forest_complet.ipynb",
        "icon": "📱",
        "title": "Téléphones : état de l'annonce",
        "description": "Prédit l'état d'un téléphone (D'occasion, Neuf, Réconditionné ou Venant) "
                       "à partir de son prix, de sa marque, de son adresse et de ses caractéristiques.",
        "task": "classification",
        "target": "etat",
        "target_label": "État prédit",
        "numeric": {
            "prix": {"type": "int", "min": 1, "max": 3_335_000, "value": 362_043, "step": 10_000},
            "dim_ecr": {"type": "float", "min": 0.0, "max": 75.0, "value": 6.0, "step": 0.1},
            "ram": {"type": "int", "min": 1, "max": 512, "value": 12, "step": 1},
            "stockage": {"type": "int", "min": 1, "max": 5120, "value": 158, "step": 1},
        },
    },
    {
        "key": "voitures_dakar",
        "prefix": "voiture",
        "notebook": "Model_for_car_data_complet.ipynb",
        "icon": "🚗",
        "title": "Voitures : Venant ou D'occasion",
        "description": "Prédit si une voiture de Dakar est **Venant** ou **D'occasion** "
                       "à partir de sa marque, de son année, de sa boîte de vitesse, de son adresse et de son prix.",
        "task": "classification",
        "target": "Venant_Occasion",
        "target_label": "Catégorie prédite",
        "preprocess": nettoyer_adresse,
        "numeric": {
            "Année": {"type": "int", "min": 1990, "max": 2024, "value": 2015, "step": 1},
            "Prix": {"type": "int", "min": 1, "max": 779_414_131, "value": 7_800_000, "step": 100_000},
        },
    },
    {
        "key": "prix_voitures",
        "prefix": "prix_voiture",
        "notebook": "Reg_Car_dataset_complet.ipynb",
        "icon": "💰",
        "title": "Voitures : prix de vente",
        "description": "Prédit le prix de vente d'une voiture à partir de son kilométrage, de son prix actuel, "
                       "de son âge, de son carburant, du type de vendeur et de sa boîte de vitesse.",
        "task": "regression",
        "target": "Selling_Price",
        "target_label": "Prix de vente prédit",
        "numeric": {
            "Kms_Driven": {"type": "int", "min": 3_000, "max": 135_000, "value": 71_000, "step": 1_000},
            "Present_Price": {"type": "float", "min": 0.3, "max": 32.2, "value": 15.2, "step": 0.5},
            "Age": {"type": "int", "min": 2, "max": 16, "value": 8, "step": 1},
        },
    },
]


# ----------------------------------------------------------------------
# Chargement des fichiers sauvegardés par les notebooks
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def charger_artefacts(prefix):
    """Retourne (artefacts, fichiers_manquants, probleme)."""
    noms = ["model", "encoders", "scaler", "uniques", "features"]
    manquants = [f"{prefix}_{n}.joblib" for n in noms if not (MODELS_DIR / f"{prefix}_{n}.joblib").exists()]
    if manquants:
        return None, manquants, None
    try:
        with warnings.catch_warnings(record=True) as avertissements:
            warnings.simplefilter("always")
            art = {n: jb.load(MODELS_DIR / f"{prefix}_{n}.joblib") for n in noms}
    except Exception as e:  # fichier illisible : le plus souvent, version de scikit-learn différente
        return None, [], f"{type(e).__name__} : {e}"
    # version de scikit-learn utilisée à l'entraînement (différente de celle installée ?)
    versions = sorted({w.message.original_sklearn_version for w in avertissements
                       if issubclass(w.category, InconsistentVersionWarning)})
    probleme = None
    if versions:
        probleme = (f"Les modèles ont été entraînés avec scikit-learn {', '.join(versions)}, mais la version "
                    f"installée est {sklearn.__version__}. Mettez `scikit-learn=={versions[0]}` dans "
                    "requirements.txt, enregistrez, puis redémarrez l'application.")
    return art, [], probleme


# ----------------------------------------------------------------------
# Prédiction (utilisée pour une ligne comme pour un fichier complet)
# ----------------------------------------------------------------------
def predire(df_in, art, cfg):
    """
    Applique les mêmes étapes que dans le notebook : encodage -> scaler -> modèle.
    Retourne (tableau avec prédictions, problèmes rencontrés, probabilités ou None).
    Les lignes avec une valeur inconnue ou manquante ne sont pas prédites (NaN).
    """
    features, encoders, uniques = art["features"], art["encoders"], art["uniques"]
    df_in = df_in.reset_index(drop=True)
    df = cfg["preprocess"](df_in) if cfg.get("preprocess") else df_in.copy()

    absentes = [c for c in features if c not in df.columns]
    if absentes:
        raise ValueError("Colonnes manquantes dans le fichier : " + ", ".join(absentes))

    X = pd.DataFrame(index=df.index, columns=features, dtype=float)
    valide = pd.Series(True, index=df.index)
    problemes = {}

    for col in features:
        if col in uniques:  # variable catégorielle -> encodeur sauvegardé
            connue = df[col].isin(encoders[col].classes_)
            if connue.any():
                X.loc[connue, col] = encoders[col].transform(df.loc[connue, col].astype(str))
            if (~connue).any():
                exemples = df.loc[~connue, col].dropna().astype(str).unique()[:5].tolist()
                problemes[col] = (int((~connue).sum()), exemples)
            valide &= connue
        else:  # variable numérique
            nombres = pd.to_numeric(df[col], errors="coerce")
            X[col] = nombres
            if nombres.isna().any():
                problemes[col] = (int(nombres.isna().sum()), ["valeur absente ou non numérique"])
            valide &= nombres.notna()

    out = df_in.copy()
    nom_pred = f"{cfg['target']}_prédit"
    out[nom_pred] = np.nan
    proba = None

    if valide.any():
        Xs = art["scaler"].transform(X.loc[valide, features].to_numpy(dtype=float))
        pred = art["model"].predict(Xs)
        index_ok = X.index[valide]

        if cfg["task"] == "classification":
            enc_cible = encoders[cfg["target"]]
            out[nom_pred] = pd.Series(enc_cible.inverse_transform(pred.astype(int)), index=index_ok)
            if hasattr(art["model"], "predict_proba"):
                proba = pd.DataFrame(art["model"].predict_proba(Xs), columns=enc_cible.classes_, index=index_ok)
        else:
            out[nom_pred] = pd.Series(pred, index=index_ok).round(2)

    return out, problemes, proba


def evaluer(out, cfg):
    """Si le fichier contient la colonne cible, compare les prédictions aux vraies valeurs."""
    cible, nom_pred = cfg["target"], f"{cfg['target']}_prédit"
    if cible not in out.columns:
        return None
    ok = out[nom_pred].notna()
    if cfg["task"] == "classification":
        if not ok.any():
            return None
        return {"Accuracy": f"{accuracy_score(out.loc[ok, cible].astype(str), out.loc[ok, nom_pred].astype(str)):.3f}"}
    reel = pd.to_numeric(out[cible], errors="coerce")
    ok = ok & reel.notna()
    if not ok.any():
        return None
    return {
        "R²": f"{r2_score(reel[ok], out.loc[ok, nom_pred]):.3f}",
        "Erreur absolue moyenne": f"{mean_absolute_error(reel[ok], out.loc[ok, nom_pred]):.2f}",
    }


def lignes_hors_plage(df, cfg):
    """Nombre de lignes dont une variable numérique sort de la plage vue à l'entraînement."""
    masque = pd.Series(False, index=df.index)
    for col, spec in cfg["numeric"].items():
        if col in df.columns:
            valeurs = pd.to_numeric(df[col], errors="coerce")
            masque |= (valeurs < spec["min"]) | (valeurs > spec["max"])
    return int(masque.sum())


def lire_fichier(fichier):
    """Lit un fichier CSV (séparateur , ou ;) ou Excel."""
    if fichier.name.lower().endswith(".xlsx"):
        return pd.read_excel(fichier)
    for encodage in ("utf-8-sig", "latin-1"):
        try:
            fichier.seek(0)
            df = pd.read_csv(fichier, encoding=encodage)
            if df.shape[1] == 1 and ";" in str(df.columns[0]):  # CSV français séparé par des ;
                fichier.seek(0)
                df = pd.read_csv(fichier, encoding=encodage, sep=";")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("Impossible de lire ce fichier CSV.")


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
def champ_numerique(label, spec, key):
    mini, maxi = spec.get("min"), spec.get("max")
    if spec.get("type", "float") == "int":
        return st.number_input(label, min_value=mini, max_value=maxi, value=int(spec["value"]),
                               step=int(spec.get("step", 1)), key=key)
    return st.number_input(label, min_value=None if mini is None else float(mini),
                           max_value=None if maxi is None else float(maxi), value=float(spec["value"]),
                           step=float(spec.get("step", 0.1)), key=key)


def afficher_problemes(problemes):
    for col, (nb, exemples) in problemes.items():
        st.warning(f"**{col}** : {nb} ligne(s) ignorée(s) ({', '.join(exemples)}).")


def onglet_unitaire(cfg, art):
    nom_pred = f"{cfg['target']}_prédit"
    with st.form(f"form_{cfg['key']}"):
        colonnes = st.columns(2)
        valeurs = {}
        for i, f in enumerate(art["features"]):
            with colonnes[i % 2]:
                label, key = f.replace("_", " "), f"{cfg['key']}_{f}"
                if f in art["uniques"]:
                    valeurs[f] = st.selectbox(label, art["uniques"][f], key=key)
                else:
                    spec = cfg["numeric"].get(f, {"type": "float", "value": 0.0, "step": 1.0})
                    valeurs[f] = champ_numerique(label, spec, key)
        soumis = st.form_submit_button("Prédire")

    if not soumis:
        return
    try:
        out, problemes, proba = predire(pd.DataFrame([valeurs]), art, cfg)
    except ValueError as e:
        st.error(str(e))
        return
    if pd.isna(out.loc[0, nom_pred]):
        st.error("Prédiction impossible avec ces valeurs.")
        afficher_problemes(problemes)
    elif cfg["task"] == "classification":
        st.markdown(
            f'<div style="background:rgba(16,185,129,0.12);border:1px solid #10b981;'
            f'border-radius:14px;padding:1.2rem 1.6rem;margin:0.6rem 0;">'
            f'<span style="color:#94a3b8;font-size:0.85rem;text-transform:uppercase;'
            f'letter-spacing:1px;">{cfg["target_label"]}</span><br>'
            f'<span style="color:#34d399;font-size:1.7rem;font-weight:700;">'
            f'{out.loc[0, nom_pred]}</span></div>',
            unsafe_allow_html=True,
        )
        if proba is not None:
            st.caption("Probabilité estimée par le modèle pour chaque classe")
            st.bar_chart(proba.iloc[0])
    else:
        st.markdown(
            f'<div style="background:rgba(59,130,246,0.12);border:1px solid #3b82f6;'
            f'border-radius:14px;padding:1.2rem 1.6rem;margin:0.6rem 0;">'
            f'<span style="color:#94a3b8;font-size:0.85rem;text-transform:uppercase;'
            f'letter-spacing:1px;">{cfg["target_label"]}</span><br>'
            f'<span style="color:#60a5fa;font-size:1.7rem;font-weight:700;">'
            f'{out.loc[0, nom_pred]:,.2f}</span></div>',
            unsafe_allow_html=True,
        )


def onglet_fichier(cfg, art):
    nom_pred = f"{cfg['target']}_prédit"
    st.info("Colonnes attendues : " + ", ".join(art["features"])
            + f". Si le fichier contient aussi la colonne `{cfg['target']}`, les prédictions seront comparées aux vraies valeurs.")
    fichier = st.file_uploader("Choisissez un fichier CSV ou Excel", type=["csv", "xlsx"], key=f"up_{cfg['key']}")
    if fichier is None:
        return
    try:
        df = lire_fichier(fichier)
        out, problemes, _ = predire(df, art, cfg)
    except ValueError as e:
        st.error(str(e))
        return

    nb_ok = int(out[nom_pred].notna().sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes du fichier", len(out))
    c2.metric("Prédictions faites", nb_ok)
    c3.metric("Lignes ignorées", len(out) - nb_ok)
    afficher_problemes(problemes)
    hors = lignes_hors_plage(df, cfg)
    if hors:
        st.warning(f"{hors} ligne(s) ont une valeur numérique en dehors de la plage vue à l'entraînement "
                   f"({', '.join(cfg['numeric'])}) : leurs prédictions sont peu fiables.")

    resultats = evaluer(out, cfg)
    if resultats:
        st.subheader(f"Comparaison avec la colonne {cfg['target']} du fichier")
        for colonne, (nom, valeur) in zip(st.columns(len(resultats)), resultats.items()):
            colonne.metric(nom, valeur)
        st.caption("Indicatif : le modèle a déjà vu une partie de ces données pendant son entraînement.")

    st.subheader("Résultats")
    st.dataframe(out.head(1000))
    if len(out) > 1000:
        st.caption("Aperçu des 1 000 premières lignes. Le fichier téléchargé contient toutes les lignes.")
    if cfg["task"] == "classification" and nb_ok:
        st.caption("Répartition des prédictions")
        st.bar_chart(out[nom_pred].value_counts())
    st.download_button("Télécharger les prédictions (CSV)", out.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"predictions_{cfg['key']}.csv", mime="text/csv")


def afficher_page(cfg):
    st.markdown(
        f'<div class="page-header"><h1>{cfg["icon"]} {cfg["title"]}</h1>'
        f"<p>{cfg['description']}</p></div>",
        unsafe_allow_html=True,
    )
    art, manquants, probleme = charger_artefacts(cfg["prefix"])
    if art is None:
        if manquants:
            st.error("Fichiers introuvables dans le dossier `models/` : " + ", ".join(manquants)
                     + f". Exécutez le notebook `{cfg['notebook']}` puis copiez ces fichiers dans `models/`.")
        else:
            st.error(f"Impossible de charger les modèles ({probleme}). Vérifiez que la version de "
                     "scikit-learn dans requirements.txt est celle utilisée dans les notebooks.")
        st.stop()
    if probleme:
        st.warning(probleme)
    onglet1, onglet2 = st.tabs(["Prédiction unitaire", "Prédiction sur un jeu de données complet"])
    with onglet1:
        onglet_unitaire(cfg, art)
    with onglet2:
        onglet_fichier(cfg, art)


def main():
    st.set_page_config(page_title="ML Predict Studio", page_icon="🤖", layout="wide",
                       initial_sidebar_state="expanded")
    injecter_style()
    pages = [
        st.Page(lambda cfg=cfg: afficher_page(cfg), title=cfg["title"], icon=cfg["icon"],
                url_path=cfg["key"], default=(i == 0))
        for i, cfg in enumerate(PAGES)
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
