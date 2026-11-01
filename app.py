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
from pathlib import Path

import joblib as jb
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

MODELS_DIR = Path(__file__).parent / "models"


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
    """Retourne (artefacts, fichiers_manquants)."""
    noms = ["model", "encoders", "scaler", "uniques", "features"]
    manquants = [f"{prefix}_{n}.joblib" for n in noms if not (MODELS_DIR / f"{prefix}_{n}.joblib").exists()]
    if manquants:
        return None, manquants
    return {n: jb.load(MODELS_DIR / f"{prefix}_{n}.joblib") for n in noms}, []


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
                out["Confiance (%)"] = (proba.max(axis=1) * 100).round(1)
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
        st.success(f"{cfg['target_label']} : **{out.loc[0, nom_pred]}**")
        if proba is not None:
            st.caption("Probabilité estimée par le modèle pour chaque classe")
            st.bar_chart(proba.iloc[0])
    else:
        st.metric(cfg["target_label"], f"{out.loc[0, nom_pred]:,.2f}")


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
    st.title(f"{cfg['icon']} {cfg['title']}")
    st.markdown(cfg["description"])
    art, manquants = charger_artefacts(cfg["prefix"])
    if art is None:
        st.error("Fichiers introuvables dans le dossier `models/` : " + ", ".join(manquants)
                 + f". Exécutez le notebook `{cfg['notebook']}` puis copiez ces fichiers dans `models/`.")
        st.stop()
    onglet1, onglet2 = st.tabs(["Prédiction unitaire", "Prédiction sur un jeu de données complet"])
    with onglet1:
        onglet_unitaire(cfg, art)
    with onglet2:
        onglet_fichier(cfg, art)


def main():
    st.set_page_config(page_title="Prédictions ML", page_icon="🤖", layout="wide")
    pages = [
        st.Page(lambda cfg=cfg: afficher_page(cfg), title=cfg["title"], icon=cfg["icon"],
                url_path=cfg["key"], default=(i == 0))
        for i, cfg in enumerate(PAGES)
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
