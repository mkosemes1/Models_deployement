import warnings                     
from pathlib import Path             

import joblib as jb                  
import numpy as np                   
import pandas as pd                 
import sklearn                       
import streamlit as st              
from sklearn.exceptions import InconsistentVersionWarning      
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score  

# Dossier models/ situé à côté de app.py : les fichiers y sont nommés <préfixe>_model.joblib, etc.
MODELS_DIR = Path(__file__).parent / "models"


# 2. DESIGN : feuille de style CSS (couleurs, cartes, boutons, onglets...)


CSS = """
<style>
    /* ================= FOND & POLICES : dégradé sombre de la page, police, style de la barre latérale et des titres ================= */
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(37,99,235,0.18), transparent 55%),
            radial-gradient(900px 500px at 110% 10%, rgba(124,58,237,0.14), transparent 50%),
            linear-gradient(180deg, #0b1220 0%, #0f172a 55%, #111827 100%);
        background-attachment: fixed;
    }
    html, body, [class*="css"], .stApp, .block-container {
        font-family: 'Segoe UI', 'Inter', system-ui, -apple-system, sans-serif !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1a 0%, #0d1526 100%) !important;
        border-right: 1px solid rgba(51, 65, 85, 0.7);
    }
    h1, h2, h3, h4 {
        color: #f1f5f9 !important;
        letter-spacing: -0.4px;
        font-weight: 700 !important;
    }
    p, li, .stMarkdown, label, span {
        color: #cbd5e1 !important;
    }
    /* ================= SIDEBAR / MARQUE : avatar, nom et badge affichés en haut de la barre latérale ================= */
    .brand {
        text-align: center;
        padding: 1.4rem 1rem 0.6rem;
    }
    .brand .avatar-wrap {
        width: 92px; height: 92px; margin: 0 auto 0.7rem;
        border-radius: 50%; padding: 3px;
        background: conic-gradient(from 210deg, #38bdf8, #818cf8, #c084fc, #38bdf8);
        box-shadow: 0 8px 28px rgba(99, 102, 241, 0.45);
    }
    .brand .avatar-wrap img {
        width: 100%; height: 100%; border-radius: 50%;
        object-fit: cover; border: 3px solid #0a0f1a; display: block;
    }
    .brand .brand-name {
        font-size: 1.35rem; font-weight: 800; margin: 0;
        background: linear-gradient(90deg, #38bdf8 0%, #a78bfa 55%, #f472b6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: 0.2px;
    }
    .brand .brand-sub {
        font-size: 0.78rem; color: #64748b !important; margin-top: 4px;
        text-transform: uppercase; letter-spacing: 1.6px;
    }
    .brand .brand-badge {
        display: inline-block; margin-top: 10px; padding: 4px 12px;
        font-size: 0.7rem; font-weight: 600; color: #93c5fd !important;
        background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.4);
        border-radius: 999px; letter-spacing: 0.6px;
    }
    .divider {
        border: none; border-top: 1px solid rgba(51,65,85,0.7);
        margin: 1.1rem 0.4rem;
    }
    /* ================= HEADER DE PAGE : bandeau coloré avec le titre et la description de chaque page ================= */
    .page-header {
        position: relative;
        background: linear-gradient(100deg, rgba(37,99,235,0.28) 0%, rgba(124,58,237,0.16) 55%, rgba(236,72,153,0.10) 100%);
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: 18px;
        padding: 1.6rem 2rem 1.5rem;
        margin-bottom: 1.6rem;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    }
    .page-header::before {
        content: ""; position: absolute; top: -40%; right: -8%;
        width: 260px; height: 260px; border-radius: 50%;
        background: radial-gradient(circle, rgba(124,58,237,0.35), transparent 65%);
        pointer-events: none;
    }
    .page-header h1 {
        margin: 0 0 0.45rem 0 !important;
        font-size: 2rem !important;
        text-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }
    .page-header p {
        margin: 0 !important; color: #a5b4c8 !important;
        font-size: 1rem; line-height: 1.55; max-width: 820px;
    }
    /* ================= FORMULAIRE : carte du formulaire de prédiction unitaire et style des champs ================= */
    [data-testid="stForm"] {
        background: linear-gradient(180deg, rgba(30,41,59,0.92) 0%, rgba(23,32,51,0.95) 100%);
        border: 1px solid rgba(71,85,105,0.8);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        box-shadow: 0 16px 48px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
    }
    [data-testid="stForm"] label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        letter-spacing: 0.2px;
    }
    /* ---- Champs plus nets et lisibles ---- */
    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div {
        background: #0d1424 !important;
        border: 1.5px solid #475569 !important;
        border-radius: 12px !important;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.35);
        transition: border-color 0.18s ease, box-shadow 0.18s ease;
    }
    [data-baseweb="input"] > div:hover,
    [data-baseweb="select"] > div:hover {
        border-color: #64748b !important;
    }
    [data-baseweb="input"] > div:focus-within,
    [data-baseweb="select"] > div:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.25), inset 0 2px 6px rgba(0,0,0,0.35) !important;
    }
    [data-baseweb="input"] input, [data-baseweb="select"] input {
        color: #f8fafc !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        caret-color: #38bdf8;
    }
    [data-baseweb="input"] input::placeholder { color: #64748b !important; font-weight: 400; }
    [data-baseweb="select"] [aria-label] { color: #f8fafc !important; font-weight: 600; }
    /* Steppers des number inputs */
    [data-baseweb="input"] button {
        background: #1e293b !important; color: #cbd5e1 !important;
        border: 1px solid #334155 !important; border-radius: 8px !important;
    }
    [data-baseweb="input"] button:hover { background: #334155 !important; color: #fff !important; }
    /* ================= BOUTONS : bouton « Prédire » et bouton de téléchargement ================= */
    [data-testid="stFormSubmitButton"] > button,
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #2563eb 0%, #6d28d9 55%, #a21caf 100%) !important;
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.4px;
        padding: 0.6rem 2rem !important;
        transition: transform 0.16s ease, box-shadow 0.16s ease, filter 0.16s ease;
        box-shadow: 0 6px 20px rgba(37,99,235,0.45);
        text-transform: uppercase;
        font-size: 0.85rem !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.12);
        box-shadow: 0 10px 28px rgba(124,58,237,0.55);
    }
    /* ================= MÉTRIQUES : cartes des compteurs (lignes, prédictions, R², ...) ================= */
    [data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
        border: 1px solid rgba(71,85,105,0.8);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; font-size: 0.75rem !important; }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 800 !important; font-size: 1.7rem !important; }
    /* ================= ONGLETS : « Prédiction unitaire » / « Jeu de données complet » ================= */
    [data-baseweb="tab-list"] {
        gap: 8px; border-bottom: 1px solid rgba(51,65,85,0.8);
        padding-bottom: 4px;
    }
    [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 12px 12px 0 0 !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        padding: 10px 26px !important;
        transition: color 0.15s ease, background 0.15s ease;
    }
    [data-baseweb="tab"]:hover { color: #cbd5e1 !important; background: rgba(51,65,85,0.3) !important; }
    [aria-selected="true"] {
        background: rgba(37,99,235,0.12) !important;
        color: #93c5fd !important;
        border-bottom: 2px solid rgba(56,189,248,0.55) !important;
    }
    /* ================= MESSAGES : couleurs des alertes (erreur, avertissement, information) ================= */
    [data-testid="stError"] {
        background: rgba(239,68,68,0.14); border-left: 4px solid #ef4444;
        border-radius: 12px; color: #fecaca !important;
        border-top: 1px solid rgba(239,68,68,0.25); border-right: 1px solid rgba(239,68,68,0.25); border-bottom: 1px solid rgba(239,68,68,0.25);
    }
    [data-testid="stWarning"] {
        background: rgba(245,158,11,0.14); border-left: 4px solid #f59e0b;
        border-radius: 12px; color: #fde68a !important;
        border-top: 1px solid rgba(245,158,11,0.25); border-right: 1px solid rgba(245,158,11,0.25); border-bottom: 1px solid rgba(245,158,11,0.25);
    }
    [data-testid="stInfo"] {
        background: rgba(59,130,246,0.14); border-left: 4px solid #3b82f6;
        border-radius: 12px; color: #bfdbfe !important;
        border-top: 1px solid rgba(59,130,246,0.25); border-right: 1px solid rgba(59,130,246,0.25); border-bottom: 1px solid rgba(59,130,246,0.25);
    }
    /* ================= TABLEAUX / UPLOAD : tableau de résultats et zone de dépôt de fichier ================= */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(71,85,105,0.8); border-radius: 14px; overflow: hidden;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    [data-testid="stFileUploader"] {
        background: linear-gradient(180deg, rgba(30,41,59,0.85), rgba(15,23,42,0.9));
        border: 2px dashed #475569 !important;
        border-radius: 16px; padding: 1.4rem;
        transition: border-color 0.15s ease;
    }
    [data-testid="stFileUploader"]:hover { border-color: #38bdf8 !important; }
    /* ================= RÉSULTAT PRÉDICTION : grande carte qui affiche la valeur prédite ================= */
    .pred-card {
        border-radius: 16px; padding: 1.4rem 1.8rem; margin: 0.8rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        border: 1px solid;
        animation: pop 0.25s ease;
    }
    @keyframes pop { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
    .pred-card .pred-label {
        font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.4px;
        color: #94a3b8 !important; font-weight: 600;
    }
    .pred-card .pred-value { font-size: 2rem; font-weight: 800; margin-top: 4px; }
    .pred-ok { background: linear-gradient(120deg, rgba(16,185,129,0.18), rgba(6,95,70,0.12)); border-color: rgba(16,185,129,0.55); }
    .pred-ok .pred-value { color: #34d399; text-shadow: 0 0 24px rgba(52,211,153,0.4); }
    .pred-info { background: linear-gradient(120deg, rgba(59,130,246,0.18), rgba(37,99,235,0.10)); border-color: rgba(59,130,246,0.55); }
    .pred-info .pred-value { color: #60a5fa; text-shadow: 0 0 24px rgba(96,165,250,0.4); }
    /* ================= FOOTER : ligne de bas de page ================= */
    .footer {
        text-align: center; color: #475569 !important; font-size: 0.8rem;
        padding: 2.2rem 0 1.2rem; border-top: 1px solid rgba(30,41,59,0.9);
        margin-top: 2.8rem; letter-spacing: 0.4px;
    }
    .footer b { color: #64748b !important; }
</style>
"""


def injecter_style():
    """Injecte le CSS, puis affiche l'en-tête de la barre latérale et le pied de page."""
    # 1) applique la feuille de style à toute l'application
    st.markdown(CSS, unsafe_allow_html=True)
    # 2) barre latérale : avatar, nom de l'application et badge
    with st.sidebar:
        st.markdown(
            '<div class="brand">'
            '<div class="avatar-wrap">'
            '<img src="https://github.com/mkosemes1.png" alt="Souane" '
            'onerror="this.src=\'https://github.com/identicons/mkosemes1.png\'">'
            "</div>"
            '<p class="brand-name">Souane\'s Models</p>'
            '<p class="brand-sub">Machine Learning</p>'
            '<span class="brand-badge">🚀 4 modèles déployés</span>'
            "</div>"
            "<hr class='divider'>",
            unsafe_allow_html=True,
        )
    # 3) pied de page
    st.markdown(
        '<div class="footer"><b>Souane\'s Models</b> — Propulsé par Streamlit &amp; scikit-learn</div>',
        unsafe_allow_html=True,
    )


# ======================================================================
# 3. NETTOYAGE DES DONNÉES ENVOYÉES (page « Voitures de Dakar » uniquement)
# ======================================================================
def nettoyer_adresse(df):
    """
    Applique à la colonne « Adresse » EXACTEMENT le même nettoyage que le notebook
    Model_for_car_data_complet.ipynb.

    Pourquoi c'est nécessaire : le modèle a été entraîné avec des adresses nettoyées (ex. « Almadies »).
    Un fichier brut contient « Almadies,Dakar » ou « Almadies, Dakar » : sans ce nettoyage,
    l'encodeur ne reconnaîtrait aucune adresse et toutes les lignes seraient ignorées.
    Si vous changez le nettoyage dans le notebook, changez-le aussi ici.
    """
    df = df.copy()  # on ne modifie pas le fichier d'origine
    if "Adresse" in df.columns:
        df["Adresse"] = (
            df["Adresse"].astype("string")
            .str.replace(r"\s*,\s*", ",", regex=True)         # 1) espaces autour de la virgule
            .str.strip()                                       # 1) espaces en début et fin de texte
            .replace({"Lac rose,Dakar": "Lac Rose,Dakar"})     # 2) même adresse écrite avec deux majuscules
            .str.split(",").str[0]                             # 3) on garde le quartier : « Almadies,Dakar » -> « Almadies »
        )
    return df


# ======================================================================
# 4. CONFIGURATION DES 4 PAGES
#    key / title / icon / description : identité de la page (URL, titre, texte d'introduction)
#    prefix   : début du nom des fichiers .joblib de la page (ex. « taxi » -> models/taxi_model.joblib)
#    notebook : notebook à exécuter si les fichiers sont absents (affiché dans le message d'erreur)
#    task     : « classification » (prédit une catégorie) ou « regression » (prédit un nombre)
#    target   : nom de la variable cible ; target_label : texte affiché au-dessus du résultat
#    preprocess : fonction de nettoyage appliquée avant la prédiction (optionnel)
#    numeric  : pour chaque variable numérique, type, plage et valeur par défaut du champ de saisie.
#               Les plages = celles vues par le modèle à l'entraînement : à mettre à jour
#               si vous relancez un notebook qui modifie les données (traitement des valeurs aberrantes).
# ======================================================================
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
            "prix": {"type": "int", "min": 1, "max": 795_000, "value": 362_043, "step": 10_000},
            "dim_ecr": {"type": "float", "min": 5.0, "max": 8.0, "value": 6.0, "step": 0.1},
            "ram": {"type": "int", "min": 7, "max": 16, "value": 12, "step": 1},
            "stockage": {"type": "int", "min": 1, "max": 348, "value": 158, "step": 1},
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
            "Année": {"type": "int", "min": 2009, "max": 2020, "value": 2015, "step": 1},
            "Prix": {"type": "int", "min": 1, "max": 24_000_000, "value": 7_800_000, "step": 100_000},
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
    {
        "key": "taxi",
        "prefix": "taxi",
        "notebook": "Reg_Mul_Reg_Ridge_Dec_Tree_Random_Forest.ipynb",
        "icon": "🚕",
        "title": "Taxi : prix du trajet",
        "description": "Prédit le prix d'un trajet de taxi à partir de la distance, du moment de la journée, "
                       "du jour, du nombre de passagers, du trafic, de la météo et des tarifs.",
        "task": "regression",
        "target": "Trip_Price",
        "target_label": "Prix du trajet prédit",
        "numeric": {
            "Trip_Distance_km": {"type": "float", "min": 1.23, "max": 49.99, "value": 25.8, "step": 0.1},
            "Passenger_Count": {"type": "int", "min": 1, "max": 4, "value": 2, "step": 1},
            "Base_Fare": {"type": "float", "min": 2.01, "max": 5.0, "value": 3.5, "step": 0.01},
            "Per_Km_Rate": {"type": "float", "min": 0.5, "max": 2.0, "value": 0.8, "step": 0.01},
            "Per_Minute_Rate": {"type": "float", "min": 0.1, "max": 0.5, "value": 0.3, "step": 0.01},
            "Trip_Duration_Minutes": {"type": "float", "min": 5.01, "max": 119.84, "value": 30.0, "step": 0.1},
        },
    },
]


# ======================================================================
# 5. CHARGEMENT DES FICHIERS SAUVEGARDÉS PAR LES NOTEBOOKS
# ======================================================================
@st.cache_resource(show_spinner=False)   # chargé une seule fois puis gardé en mémoire (application plus rapide)
def charger_artefacts(prefix):
    """Charge les 5 fichiers d'un modèle. Retourne (artefacts, fichiers_manquants, probleme)."""
    # model = le modèle ; encoders = les encodeurs des variables catégorielles ; scaler = la normalisation ;
    # uniques = les valeurs des listes déroulantes ; features = l'ordre des variables
    noms = ["model", "encoders", "scaler", "uniques", "features"]
    manquants = [f"{prefix}_{n}.joblib" for n in noms if not (MODELS_DIR / f"{prefix}_{n}.joblib").exists()]
    if manquants:
        return None, manquants, None
    try:
        with warnings.catch_warnings(record=True) as avertissements:   # on capture les avertissements de chargement
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


# ======================================================================
# 6. PRÉDICTION (utilisée pour une seule ligne comme pour un fichier complet)
# ======================================================================
def predire(df_in, art, cfg):
    """
    Refait les mêmes étapes que dans le notebook : nettoyage -> encodage -> scaler -> modèle.
    Retourne (tableau avec la colonne prédite, problèmes rencontrés, probabilités ou None).
    Les lignes avec une valeur inconnue ou manquante ne sont pas prédites (NaN).
    """
    features, encoders, uniques = art["features"], art["encoders"], art["uniques"]
    df_in = df_in.reset_index(drop=True)
    # nettoyage propre à la page (ex. adresses), sinon simple copie
    df = cfg["preprocess"](df_in) if cfg.get("preprocess") else df_in.copy()

    # le fichier doit contenir toutes les colonnes utilisées par le modèle
    absentes = [c for c in features if c not in df.columns]
    if absentes:
        raise ValueError("Colonnes manquantes dans le fichier : " + ", ".join(absentes))

    X = pd.DataFrame(index=df.index, columns=features, dtype=float)   # tableau des valeurs numériques finales
    valide = pd.Series(True, index=df.index)                           # False = ligne impossible à prédire
    problemes = {}

    for col in features:
        if col in uniques:  # variable catégorielle -> on applique l'encodeur sauvegardé
            connue = df[col].isin(encoders[col].classes_)   # valeur connue de l'encodeur ?
            if connue.any():
                X.loc[connue, col] = encoders[col].transform(df.loc[connue, col].astype(str))
            if (~connue).any():
                exemples = df.loc[~connue, col].dropna().astype(str).unique()[:5].tolist()
                problemes[col] = (int((~connue).sum()), exemples)
            valide &= connue
        else:  # variable numérique -> conversion en nombres
            nombres = pd.to_numeric(df[col], errors="coerce")
            X[col] = nombres
            if nombres.isna().any():
                problemes[col] = (int(nombres.isna().sum()), ["valeur absente ou non numérique"])
            valide &= nombres.notna()

    out = df_in.copy()                       # on garde les colonnes d'origine du fichier
    nom_pred = f"{cfg['target']}_prédit"
    out[nom_pred] = np.nan
    proba = None

    if valide.any():
        # normalisation avec le scaler sauvegardé, puis prédiction du modèle
        Xs = art["scaler"].transform(X.loc[valide, features].to_numpy(dtype=float))
        pred = art["model"].predict(Xs)
        index_ok = X.index[valide]

        if cfg["task"] == "classification":
            enc_cible = encoders[cfg["target"]]
            # on retrouve le texte de la classe (ex. 0 -> « D'occasion ») avec l'encodeur de la cible
            out[nom_pred] = pd.Series(enc_cible.inverse_transform(pred.astype(int)), index=index_ok)
            # probabilité de chaque classe (affichée dans le graphique de la prédiction unitaire)
            proba = pd.DataFrame(art["model"].predict_proba(Xs), columns=enc_cible.classes_, index=index_ok)
        else:
            out[nom_pred] = pd.Series(pred, index=index_ok).round(2)

    return out, problemes, proba


def evaluer(out, cfg):
    """Si le fichier contient la colonne cible, compare les prédictions aux vraies valeurs."""
    cible, nom_pred = cfg["target"], f"{cfg['target']}_prédit"
    if cible not in out.columns:
        return None
    ok = out[nom_pred].notna()   # seulement les lignes réellement prédites
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
    """Compte les lignes dont une variable numérique sort de la plage vue à l'entraînement (prédictions peu fiables)."""
    masque = pd.Series(False, index=df.index)
    for col, spec in cfg["numeric"].items():
        if col in df.columns:
            valeurs = pd.to_numeric(df[col], errors="coerce")
            masque |= (valeurs < spec["min"]) | (valeurs > spec["max"])
    return int(masque.sum())


# ======================================================================
# 7. LECTURE DU FICHIER ENVOYÉ PAR L'UTILISATEUR
# ======================================================================
def lire_fichier(fichier):
    """Lit un fichier CSV (séparateur , ou ;) ou Excel."""
    if fichier.name.lower().endswith(".xlsx"):
        return pd.read_excel(fichier)
    for encodage in ("utf-8-sig", "latin-1"):   # on essaie UTF-8, puis le codage des CSV Excel français
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


# ======================================================================
# 8. INTERFACE : formulaire, onglets et pages
# ======================================================================
def champ_numerique(label, spec, key):
    """Crée le champ de saisie d'une variable numérique d'après sa configuration (entier ou décimal)."""
    if spec["type"] == "int":
        return st.number_input(label, min_value=spec["min"], max_value=spec["max"],
                               value=spec["value"], step=spec["step"], key=key)
    return st.number_input(label, min_value=float(spec["min"]), max_value=float(spec["max"]),
                           value=float(spec["value"]), step=float(spec["step"]), key=key)


def afficher_problemes(problemes):
    """Affiche un avertissement par colonne qui a fait ignorer des lignes (valeur inconnue ou manquante)."""
    for col, (nb, exemples) in problemes.items():
        st.warning(f"**{col}** : {nb} ligne(s) ignorée(s) ({', '.join(exemples)}).")


def onglet_unitaire(cfg, art):
    """Onglet 1 : formulaire de prédiction pour une seule ligne."""
    nom_pred = f"{cfg['target']}_prédit"
    with st.form(f"form_{cfg['key']}"):
        colonnes = st.columns(2)   # les champs sont répartis sur 2 colonnes
        valeurs = {}
        for i, f in enumerate(art["features"]):
            with colonnes[i % 2]:
                label, key = f.replace("_", " "), f"{cfg['key']}_{f}"
                if f in art["uniques"]:   # variable catégorielle -> liste déroulante
                    valeurs[f] = st.selectbox(label, art["uniques"][f], key=key)
                else:                     # variable numérique -> champ numérique
                    valeurs[f] = champ_numerique(label, cfg["numeric"][f], key)
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
        # carte verte avec la classe prédite, puis graphique des probabilités
        st.markdown(
            f'<div class="pred-card pred-ok">'
            f'<div class="pred-label">{cfg["target_label"]}</div>'
            f'<div class="pred-value">{out.loc[0, nom_pred]}</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("Probabilité estimée par le modèle pour chaque classe")
        st.bar_chart(proba.iloc[0])
    else:
        # carte bleue avec le nombre prédit
        st.markdown(
            f'<div class="pred-card pred-info">'
            f'<div class="pred-label">{cfg["target_label"]}</div>'
            f'<div class="pred-value">{out.loc[0, nom_pred]:,.2f}</div></div>',
            unsafe_allow_html=True,
        )


def onglet_fichier(cfg, art):
    """Onglet 2 : prédictions sur un fichier complet, comparaison éventuelle et téléchargement."""
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

    # compteurs : lignes lues, lignes prédites, lignes ignorées
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

    # comparaison avec les vraies valeurs si le fichier contient la colonne cible
    resultats = evaluer(out, cfg)
    if resultats:
        st.subheader(f"Comparaison avec la colonne {cfg['target']} du fichier")
        for colonne, (nom, valeur) in zip(st.columns(len(resultats)), resultats.items()):
            colonne.metric(nom, valeur)
        st.caption("Indicatif : le modèle a déjà vu une partie de ces données pendant son entraînement.")

    # tableau des résultats, graphique et téléchargement
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
    """Affiche une page complète : bandeau de titre, chargement des fichiers, puis les 2 onglets."""
    st.markdown(
        f'<div class="page-header"><h1>{cfg["icon"]} {cfg["title"]}</h1>'
        f"<p>{cfg['description']}</p></div>",
        unsafe_allow_html=True,
    )
    art, manquants, probleme = charger_artefacts(cfg["prefix"])
    if art is None:   # fichiers absents ou illisibles : message clair puis arrêt de la page
        if manquants:
            st.error("Fichiers introuvables dans le dossier `models/` : " + ", ".join(manquants)
                     + f". Exécutez le notebook `{cfg['notebook']}` puis copiez ces fichiers dans `models/`.")
        else:
            st.error(f"Impossible de charger les modèles ({probleme}). Vérifiez que la version de "
                     "scikit-learn dans requirements.txt est celle utilisée dans les notebooks.")
        st.stop()
    if probleme:   # modèles chargés mais version de scikit-learn différente
        st.warning(probleme)
    onglet1, onglet2 = st.tabs(["Prédiction unitaire", "Prédiction sur un jeu de données complet"])
    with onglet1:
        onglet_unitaire(cfg, art)
    with onglet2:
        onglet_fichier(cfg, art)


# ======================================================================
# 9. POINT D'ENTRÉE : configuration de la page et navigation entre les 4 pages
# ======================================================================
def main():
    st.set_page_config(page_title="Souane's Models", page_icon="🧠", layout="wide",
                       initial_sidebar_state="expanded")
    injecter_style()   # design + barre latérale + pied de page
    # une page de navigation par modèle, construite à partir de la liste PAGES (la 1re est la page d'accueil)
    pages = [
        st.Page(lambda cfg=cfg: afficher_page(cfg), title=cfg["title"], icon=cfg["icon"],
                url_path=cfg["key"], default=(i == 0))
        for i, cfg in enumerate(PAGES)
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()