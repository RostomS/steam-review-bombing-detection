"""
Construction de la base de donnees SQL persistante (point 3 de la consigne).

Choix : DuckDB plutot qu'un SGBD serveur classique (PostgreSQL) ou une base
NoSQL (MongoDB). Justification : les donnees sont tabulaires et de schema fixe
(mêmes colonnes pour chaque avis), l'analyse repose sur des agregations
(comptages par jeu, par periode, correlations), ce qui correspond au cas
d'usage SQL plutot qu'a un besoin de flexibilite de schema propre au NoSQL.
DuckDB permet en plus de rester entierement local, sans compte cloud ni
quota, tout en offrant un vrai moteur SQL avec index et partitionnement.

Ce script :
1. Charge le Parquet filtre dans une base DuckDB persistante sur disque.
2. Cree des index sur les colonnes les plus interrogees (game, timestamp_created, voted_up).
3. Exporte une version partitionnee par jeu, pour illustrer et mesurer le partitionnement.
4. Compare le temps d'une requete avant et apres optimisation, pour documenter
   le critere d'evaluation "rapidite de traitement et temps de reponse".
"""

import os
import time
import duckdb

FILTERED_PARQUET_PATH = "data/steam_reviews_filtered.parquet"
DATABASE_PATH = "data/steam_reviews.duckdb"
PARTITIONED_DIR = "data/steam_reviews_partitioned"
MEMORY_LIMIT = "4GB"


def timed_query(con, label, query):
    start = time.perf_counter()
    result = con.execute(query).fetchdf()
    elapsed = time.perf_counter() - start
    print(f"{label} : {elapsed:.3f} s, {len(result)} ligne(s)")
    return result


def main():
    if not os.path.exists(FILTERED_PARQUET_PATH):
        raise FileNotFoundError(
            f"{FILTERED_PARQUET_PATH} introuvable. Lancer 02_filter_export.py avant ce script."
        )

    con = duckdb.connect(DATABASE_PATH, config={"memory_limit": MEMORY_LIMIT})

    print("Chargement du Parquet dans la table reviews...")
    con.execute(f"""
        CREATE OR REPLACE TABLE reviews AS
        SELECT * FROM read_parquet('{FILTERED_PARQUET_PATH}')
    """)

    row_count = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    print("Lignes chargees dans la base :", row_count)

    print("\nRequete de reference avant indexation :")
    timed_query(con, "Avis negatifs par jeu sur un jeu cible", """
        SELECT game, COUNT(*) AS n
        FROM reviews
        WHERE game = 'War Thunder' AND voted_up = 0
        GROUP BY game
    """)

    print("\nCreation des index...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_reviews_game ON reviews(game)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_reviews_timestamp ON reviews(timestamp_created)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_reviews_voted_up ON reviews(voted_up)")

    print("\nMeme requete apres indexation :")
    timed_query(con, "Avis negatifs par jeu sur un jeu cible (indexe)", """
        SELECT game, COUNT(*) AS n
        FROM reviews
        WHERE game = 'War Thunder' AND voted_up = false
        GROUP BY game
    """)

    print("\nExport partitionne par jeu, pour les requetes qui ciblent un jeu precis...")
    con.execute(f"""
        COPY reviews TO '{PARTITIONED_DIR}' (FORMAT PARQUET, PARTITION_BY (game), OVERWRITE_OR_IGNORE true)
    """)
    print("Partitions ecrites dans", PARTITIONED_DIR)

    con.execute(f"""
        CREATE OR REPLACE VIEW reviews_partitioned AS
        SELECT * FROM read_parquet('{PARTITIONED_DIR}/*/*.parquet', hive_partitioning=true)
    """)

    print("\nRequete equivalente sur la vue partitionnee :")
    timed_query(con, "Avis negatifs par jeu (vue partitionnee)", """
        SELECT game, COUNT(*) AS n
        FROM reviews_partitioned
        WHERE game = 'War Thunder' AND voted_up = 0
        GROUP BY game
    """)

    print("\nBase disponible dans :", DATABASE_PATH)
    print("Table principale : reviews")
    print("Vue partitionnee : reviews_partitioned")

    con.close()


if __name__ == "__main__":
    main()
