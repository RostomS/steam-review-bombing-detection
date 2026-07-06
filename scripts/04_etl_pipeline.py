"""
Pipeline ETL (point 4 de la consigne), automatise en un seul script.

Extraction : lecture de la table reviews deja chargee en base (03_build_database.py),
qui elle-meme part du Parquet filtre (02_filter_export.py). On ne reprend pas le CSV
brut ici : cette etape se concentre sur le nettoyage et la mise en forme, a partir
d'un sous-ensemble deja filtre et documente comme tel dans le rapport.

Transformation :
- Suppression des doublons sur recommendationid (identifiant unique d'avis).
- Suppression des avis sans texte exploitable (valeurs manquantes sur la colonne review).
- Conversion des timestamps Unix en dates lisibles.
- Encodage : les colonnes booleennes (voted_up, steam_purchase, received_for_free,
  written_during_early_access, hidden_in_steam_china) sont deja en 0/1, verifie et documente.
- Normalisation z-score des colonnes numeriques utilisees plus tard en ML
  (author_playtime_forever, author_num_reviews, weighted_vote_score, votes_up, votes_funny).

Chargement : ecriture du resultat dans une nouvelle table reviews_clean de la meme
base DuckDB, et export d'un Parquet autonome pour le notebook d'analyse.
"""

import os
import time
import duckdb

DATABASE_PATH = "data/steam_reviews.duckdb"
CLEAN_PARQUET_PATH = "data/steam_reviews_clean.parquet"
MEMORY_LIMIT = "4GB"


def main():
    if not os.path.exists(DATABASE_PATH):
        raise FileNotFoundError(
            f"{DATABASE_PATH} introuvable. Lancer 03_build_database.py avant ce script."
        )

    con = duckdb.connect(DATABASE_PATH, config={"memory_limit": MEMORY_LIMIT})
    start = time.perf_counter()

    rows_before = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    duplicates = con.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT recommendationid) FROM reviews
    """).fetchone()[0]
    missing_review = con.execute("""
        SELECT COUNT(*) FROM reviews WHERE review IS NULL OR trim(review) = ''
    """).fetchone()[0]

    print("Lignes avant nettoyage :", rows_before)
    print("Doublons detectes sur recommendationid :", duplicates)
    print("Avis sans texte exploitable :", missing_review)

    print("\nConstruction de reviews_clean...")
    con.execute("""
        CREATE OR REPLACE TABLE reviews_clean AS
        WITH deduplicated AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY recommendationid ORDER BY timestamp_created) AS rn
            FROM reviews
        ),
        stats AS (
            SELECT
                AVG(author_playtime_forever) AS avg_playtime, STDDEV(author_playtime_forever) AS std_playtime,
                AVG(author_num_reviews) AS avg_num_reviews, STDDEV(author_num_reviews) AS std_num_reviews,
                AVG(weighted_vote_score) AS avg_weighted_score, STDDEV(weighted_vote_score) AS std_weighted_score,
                AVG(votes_up) AS avg_votes_up, STDDEV(votes_up) AS std_votes_up,
                AVG(votes_funny) AS avg_votes_funny, STDDEV(votes_funny) AS std_votes_funny
            FROM deduplicated
            WHERE rn = 1 AND review IS NOT NULL AND trim(review) != ''
        )
        SELECT
            d.recommendationid,
            d.appid,
            d.game,
            d.author_steamid,
            d.author_num_games_owned,
            d.author_num_reviews,
            d.author_playtime_forever,
            d.author_playtime_last_two_weeks,
            d.author_playtime_at_review,
            to_timestamp(d.author_last_played) AS author_last_played_at,
            d.language,
            d.review,
            to_timestamp(d.timestamp_created) AS created_at,
            to_timestamp(d.timestamp_updated) AS updated_at,
            d.voted_up,
            d.votes_up,
            d.votes_funny,
            d.weighted_vote_score,
            d.comment_count,
            d.steam_purchase,
            d.received_for_free,
            d.written_during_early_access,
            d.hidden_in_steam_china,
            (d.author_playtime_forever - s.avg_playtime) / NULLIF(s.std_playtime, 0) AS playtime_forever_norm,
            (d.author_num_reviews - s.avg_num_reviews) / NULLIF(s.std_num_reviews, 0) AS num_reviews_norm,
            (d.weighted_vote_score - s.avg_weighted_score) / NULLIF(s.std_weighted_score, 0) AS weighted_score_norm,
            (d.votes_up - s.avg_votes_up) / NULLIF(s.std_votes_up, 0) AS votes_up_norm,
            (d.votes_funny - s.avg_votes_funny) / NULLIF(s.std_votes_funny, 0) AS votes_funny_norm
        FROM deduplicated d, stats s
        WHERE d.rn = 1 AND d.review IS NOT NULL AND trim(d.review) != ''
    """)

    rows_after = con.execute("SELECT COUNT(*) FROM reviews_clean").fetchone()[0]
    print("Lignes apres nettoyage :", rows_after)
    print("Lignes supprimees :", rows_before - rows_after)

    print("\nExport vers Parquet autonome...")
    con.execute(f"""
        COPY reviews_clean TO '{CLEAN_PARQUET_PATH}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    taille_go = os.path.getsize(CLEAN_PARQUET_PATH) / 1e9
    elapsed = time.perf_counter() - start
    print("Fichier exporte :", CLEAN_PARQUET_PATH)
    print("Taille finale :", round(taille_go, 2), "Go")
    print("Duree totale du pipeline :", round(elapsed, 2), "s")

    con.close()


if __name__ == "__main__":
    main()
