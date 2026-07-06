"""
Construction du jeu d'etiquettes pour la classification supervisee (point 6).

Principe : predire, a partir du seul texte d'un avis, si cet avis a ete ecrit
pendant un episode de review bombing confirme. L'etiquette est derivee de la
regle validee en EDA (05_eda.ipynb, sections 5 et 5.2) : anomalie de volume
(z-score > 3) ET ecart de taux d'avis negatifs marque par rapport a la
moyenne habituelle du jeu. Cette regle est etendue ici aux 506 jeux du
dataset filtre, au lieu des trois jeux utilises pour l'illustrer en EDA.

Point important : les variables ayant servi a construire l'etiquette
(z_score_volume, ecart_taux_negatif) ne doivent pas etre utilisees comme
variables d'entree du modele de classification, sous peine de fuite de
donnees (le modele retrouverait la regle plutot que d'apprendre un signal
textuel independant). Elles sont conservees dans l'export final a titre
de reference et de diagnostic uniquement.

Le seuil ECART_TAUX_NEGATIF_THRESHOLD (0.20) est choisi a partir des valeurs
observees en EDA : le cas confirme de War Thunder (mai 2023) presentait un
ecart de 0.45 a 0.62, tandis que les anomalies de volume sans lien avec une
controverse presentaient un ecart nul ou negatif. Un seuil de 0.20 reste
prudent, a mi-chemin, sans etre calibre finement jeu par jeu.

Les avis positifs (label 1, rares) sont tous conserves. Les avis negatifs
(label 0) sont sous-echantillonnes pour obtenir un jeu d'entrainement
equilibre et exploitable, plutot que de garder l'integralite des 32 millions
d'avis dont la tres grande majorite ne concerne aucun episode de bombing.
"""

import os
import duckdb

DATABASE_PATH = "data/steam_reviews.duckdb"
OUTPUT_PATH = "data/reviews_labeled.parquet"
MEMORY_LIMIT = "4GB"

Z_SCORE_THRESHOLD = 3
ECART_TAUX_NEGATIF_THRESHOLD = 0.20
NEGATIVE_TO_POSITIVE_RATIO = 5


def main():
    if not os.path.exists(DATABASE_PATH):
        raise FileNotFoundError(
            f"{DATABASE_PATH} introuvable. Lancer 03_build_database.py et 04_etl_pipeline.py avant ce script."
        )

    con = duckdb.connect(DATABASE_PATH, read_only=True, config={"memory_limit": MEMORY_LIMIT})

    print("Agregation quotidienne par jeu (506 jeux)...")
    con.execute("""
        CREATE TEMP TABLE daily AS
        SELECT
            game,
            date_trunc('day', created_at) AS jour,
            COUNT(*) AS n_avis,
            SUM(CASE WHEN voted_up = 0 THEN 1 ELSE 0 END) AS n_negatifs
        FROM reviews_clean
        GROUP BY game, jour
    """)

    print("Calcul du z-score de volume et de l'ecart de taux negatif, par jeu...")
    con.execute(f"""
        CREATE TEMP TABLE day_labels AS
        WITH stats AS (
            SELECT
                game,
                AVG(n_avis) AS mean_n_avis,
                STDDEV(n_avis) AS std_n_avis,
                AVG(CAST(n_negatifs AS DOUBLE) / n_avis) AS mean_taux_negatif
            FROM daily
            GROUP BY game
        )
        SELECT
            d.game,
            d.jour,
            d.n_avis,
            (d.n_avis - s.mean_n_avis) / NULLIF(s.std_n_avis, 0) AS z_score_volume,
            (CAST(d.n_negatifs AS DOUBLE) / d.n_avis) - s.mean_taux_negatif AS ecart_taux_negatif
        FROM daily d
        JOIN stats s ON d.game = s.game
    """)

    con.execute(f"""
        ALTER TABLE day_labels ADD COLUMN label INTEGER;
        UPDATE day_labels
        SET label = CASE
            WHEN z_score_volume > {Z_SCORE_THRESHOLD} AND ecart_taux_negatif > {ECART_TAUX_NEGATIF_THRESHOLD}
            THEN 1 ELSE 0
        END
    """)

    counts = con.execute("""
        SELECT
            COUNT(*) AS jours_total,
            SUM(CASE WHEN z_score_volume > 3 THEN 1 ELSE 0 END) AS jours_anomalie_volume,
            SUM(label) AS jours_review_bombing
        FROM day_labels
    """).fetchdf()
    print(counts)

    print("\nAffectation de l'etiquette a chaque avis individuel...")
    con.execute("""
        CREATE TEMP TABLE reviews_with_label AS
        SELECT
            r.review,
            d.label,
            r.game,
            d.jour,
            d.z_score_volume,
            d.ecart_taux_negatif
        FROM reviews_clean r
        JOIN day_labels d ON r.game = d.game AND date_trunc('day', r.created_at) = d.jour
    """)

    label_counts = con.execute("""
        SELECT label, COUNT(*) AS n FROM reviews_with_label GROUP BY label
    """).fetchdf()
    print(label_counts)

    n_positifs = int(label_counts.loc[label_counts["label"] == 1, "n"].iloc[0])
    n_negatifs_a_garder = n_positifs * NEGATIVE_TO_POSITIVE_RATIO
    print(f"\nAvis positifs (label 1) conserves en totalite : {n_positifs}")
    print(f"Avis negatifs (label 0) sous-echantillonnes a : {n_negatifs_a_garder}")

    print("\nConstruction de l'echantillon final equilibre et export...")
    con.execute(f"""
        COPY (
            SELECT * FROM reviews_with_label WHERE label = 1
            UNION ALL
            SELECT * FROM (
                SELECT * FROM reviews_with_label
                WHERE label = 0
                ORDER BY random()
                LIMIT {n_negatifs_a_garder}
            )
        ) TO '{OUTPUT_PATH}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    taille_mo = os.path.getsize(OUTPUT_PATH) / 1e6
    print("Fichier exporte :", OUTPUT_PATH)
    print("Taille :", round(taille_mo, 1), "Mo")

    con.close()


if __name__ == "__main__":
    main()
