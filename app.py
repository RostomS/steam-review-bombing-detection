"""
Interface de demonstration Streamlit (point 8 de la consigne).

Reutilise directement les artefacts deja produits par le reste du projet :
- data/steam_reviews.duckdb (table reviews_clean, produite par le pipeline ETL)
- models/tfidf_vectorizer.joblib et models/logistic_model.joblib (07_ml_supervised.ipynb)
- data/reviews_labeled.parquet, pour proposer des exemples reels plutot que des textes inventes

Lancer avec : streamlit run app.py
"""

import os
import duckdb
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

DATABASE_PATH = "data/steam_reviews.duckdb"
LABELED_DATA_PATH = "data/reviews_labeled.parquet"
MODEL_DIR = "models"

st.set_page_config(page_title="Detection de review bombing - Steam Reviews", layout="wide")


@st.cache_resource
def load_models():
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    logistic_model = joblib.load(os.path.join(MODEL_DIR, "logistic_model.joblib"))
    return vectorizer, logistic_model


@st.cache_resource
def get_connection():
    return duckdb.connect(DATABASE_PATH, read_only=True)


@st.cache_data
def load_game_list(_con):
    return _con.execute("SELECT DISTINCT game FROM reviews_clean ORDER BY game").fetchdf()["game"].tolist()


@st.cache_data
def load_example_reviews():
    df = pd.read_parquet(LABELED_DATA_PATH, columns=["review", "label", "game", "jour"])
    positifs = df[df["label"] == 1].sample(3, random_state=42)
    negatifs = df[df["label"] == 0].sample(3, random_state=42)
    return pd.concat([positifs, negatifs]).reset_index(drop=True)


vectorizer, logistic_model = load_models()
con = get_connection()

st.title("Detection de review bombing sur Steam Reviews")
st.caption("Chaine complete de data science : collecte, stockage, ETL, analyse, machine learning, demonstration")

tab_donnees, tab_prediction, tab_methode = st.tabs(
    ["Visualisation des donnees", "Prediction sur un avis", "Methodologie"]
)

with tab_donnees:
    st.subheader("Evolution quotidienne du volume d'avis par jeu")

    game_list = load_game_list(con)
    default_index = game_list.index("War Thunder") if "War Thunder" in game_list else 0
    selected_game = st.selectbox("Choisir un jeu", game_list, index=default_index)

    daily = con.execute(
        """
        SELECT
            date_trunc('day', created_at) AS jour,
            COUNT(*) AS n_avis,
            SUM(CASE WHEN voted_up = 0 THEN 1 ELSE 0 END) AS n_negatifs
        FROM reviews_clean
        WHERE game = ?
        GROUP BY jour
        ORDER BY jour
        """,
        [selected_game],
    ).fetchdf()

    if len(daily) > 5:
        daily["z_score"] = (daily["n_avis"] - daily["n_avis"].mean()) / daily["n_avis"].std()
        daily["anomalie"] = daily["z_score"] > 3
    else:
        daily["anomalie"] = False

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily["jour"], daily["n_avis"], color="steelblue", linewidth=0.8)
    flagged = daily[daily["anomalie"]]
    ax.scatter(flagged["jour"], flagged["n_avis"], color="red", zorder=5, label="Anomalie de volume (z-score > 3)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avis par jour")
    ax.set_title(f"Volume quotidien d'avis - {selected_game}")
    ax.legend()
    st.pyplot(fig)

    st.caption(
        "Methode identique a celle de l'analyse exploratoire (05_eda.ipynb) : un jour est signale comme "
        "anomalie si son volume depasse 3 ecarts-types par rapport a la moyenne du jeu selectionne."
    )

    total_avis = int(daily["n_avis"].sum()) if len(daily) > 0 else 0
    taux_negatif = (daily["n_negatifs"].sum() / total_avis) if total_avis > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total d'avis", f"{total_avis:,}".replace(",", " "))
    with col2:
        st.metric("Taux d'avis negatifs", f"{taux_negatif:.1%}")

with tab_prediction:
    st.subheader("Predire si un avis correspond a un profil de review bombing")
    st.write(
        "Modele retenu : regression logistique entrainee sur une representation TF-IDF du texte "
        "(F1 = 0.600 sur la classe review bombing, voir 07_ml_supervised.ipynb). Le modele utilise "
        "uniquement le texte de l'avis, jamais les statistiques de volume ou de negativite."
    )

    examples = load_example_reviews()
    example_labels = [
        f"Exemple {i + 1} ({'review bombing' if row['label'] == 1 else 'normal'}) - {row['game']}"
        for i, row in examples.iterrows()
    ]
    selected_example = st.selectbox(
        "Charger un exemple reel issu du jeu de donnees (optionnel)", ["Aucun"] + example_labels
    )

    default_text = ""
    if selected_example != "Aucun":
        idx = example_labels.index(selected_example)
        default_text = examples.iloc[idx]["review"]

    user_text = st.text_area("Texte de l'avis", value=default_text, height=150)

    if st.button("Predire"):
        if not user_text.strip():
            st.warning("Merci de saisir un texte avant de lancer la prediction.")
        else:
            X_input = vectorizer.transform([user_text])
            proba = logistic_model.predict_proba(X_input)[0][1]
            prediction = logistic_model.predict(X_input)[0]

            if prediction == 1:
                st.error(f"Profil de review bombing detecte (probabilite : {proba:.1%})")
            else:
                st.success(f"Avis normal (probabilite de review bombing : {proba:.1%})")

            feature_names = vectorizer.get_feature_names_out()
            input_array = X_input.toarray()[0]
            nonzero_idx = np.nonzero(input_array)[0]

            if len(nonzero_idx) > 0:
                contributions = input_array[nonzero_idx] * logistic_model.coef_[0][nonzero_idx]
                contrib_df = pd.DataFrame(
                    {"terme": feature_names[nonzero_idx], "contribution": contributions}
                ).sort_values("contribution", ascending=False)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("Termes poussant vers review bombing")
                    st.dataframe(contrib_df.head(10), hide_index=True)
                with col_b:
                    st.write("Termes poussant vers avis normal")
                    st.dataframe(contrib_df.tail(10).sort_values("contribution"), hide_index=True)
            else:
                st.info("Aucun mot de cet avis ne fait partie du vocabulaire de 20 000 termes appris par le modele.")

with tab_methode:
    st.subheader("Rappel methodologique")
    st.markdown(
        """
        - **Etiquette** : un avis est etiquete review bombing si le jour et le jeu concernes presentent a la fois
          une anomalie de volume (z-score > 3) et un ecart de taux d'avis negatifs superieur a 20 points par
          rapport a la moyenne habituelle du jeu (voir scripts/06_build_labels.py).
        - **Modele** : regression logistique sur TF-IDF (20 000 termes), sans les statistiques de volume ou de
          negativite comme variables d'entree, pour eviter que le modele ne retrouve simplement la regle
          d'etiquetage plutot que d'apprendre un signal textuel independant.
        - **Limite** : l'etiquette est derivee d'une regle statistique appliquee automatiquement aux 506 jeux du
          dataset filtre, pas d'une verification manuelle exhaustive de chaque episode.

        Detail complet de la methodologie, des resultats et des difficultes rencontrees dans le rapport
        technique (README.md).
        """
    )
