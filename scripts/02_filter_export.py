"""
Filtrage final et export Parquet, en local avec DuckDB.

TARGET_GAMES = jeux a controverse confirmes + les N jeux les plus reviewes,
d'apres le comptage produit par 01_count_games.py.

Une seule requete DuckDB fait le filtre et l'export, sans etape intermediaire
en memoire Python.
"""

import os
import duckdb
import pandas as pd

CSV_PATH = "data/steam-reviews/all_reviews/all_reviews.csv"
GAME_COUNTS_PATH = "scripts/game_counts.csv"
OUTPUT_PARQUET_PATH = "data/steam_reviews_filtered.parquet"
MEMORY_LIMIT = "4GB"

# Noms exacts confirmes par 01_count_games.py sur le dataset mis a jour (113,9M lignes)
CONTROVERSY_GAMES = [
    "War Thunder",
    "Grand Theft Auto V",
    "Crusader Kings III",
    "Crusader Kings II",
    "Counter-Strike 2",
    "Counter-Strike",
    "Counter-Strike: Source",
    "Counter-Strike: Condition Zero",
    "Dota 2",
    "Terraria",
    "Cyberpunk 2077",
    "ELDEN RING",
    "Apex Legends",
    "Destiny 2",
    "Baldur's Gate 3",
    "Fall Guys",
    "Red Dead Redemption 2",
    "New World",
    "Halo Infinite (Campaign)",
    "Hades",
    "Team Fortress 2",
    "PUBG: BATTLEGROUNDS",
    "Kerbal Space Program 2",
    "Assassin's Creed Unity",
    "Sonic Mania",
    "Firewatch",
    "Titan Souls",
    "Stardew Valley",
]

# Dataset deux fois plus gros que lors du premier essai (113,9M lignes contre 58M) :
# valeur reduite par precaution pour ne pas depasser le seuil gratuit BigQuery des le premier essai.
TOP_N_POPULAR = 500


def main():
    if not os.path.exists(GAME_COUNTS_PATH):
        raise FileNotFoundError(
            f"{GAME_COUNTS_PATH} introuvable. Lancer 01_count_games.py avant ce script."
        )

    counts = pd.read_csv(GAME_COUNTS_PATH)
    counts = counts[counts["game"].apply(lambda x: isinstance(x, str))].reset_index(drop=True)

    manquants = [g for g in CONTROVERSY_GAMES if g not in counts["game"].values]
    if manquants:
        print("Attention, noms absents du comptage, a retirer ou corriger :", manquants)

    top_popular_games = counts.head(TOP_N_POPULAR)["game"].tolist()
    target_games = sorted(set(CONTROVERSY_GAMES) | set(top_popular_games))

    print("Jeux a controverse confirmes :", len(CONTROVERSY_GAMES))
    print("Jeux populaires ajoutes :", len(top_popular_games))
    print("Total des jeux cibles :", len(target_games))

    con = duckdb.connect(config={"memory_limit": MEMORY_LIMIT})
    con.register("target_games_df", pd.DataFrame({"game": target_games}))

    con.execute(f"""
        COPY (
            SELECT r.*
            FROM read_csv_auto('{CSV_PATH}', ignore_errors=true) r
            JOIN target_games_df t ON r.game = t.game
            WHERE r.language = 'english'
        ) TO '{OUTPUT_PARQUET_PATH}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    taille_go = os.path.getsize(OUTPUT_PARQUET_PATH) / 1e9
    print("Fichier exporte :", OUTPUT_PARQUET_PATH)
    print("Taille finale :", round(taille_go, 2), "Go")
    if taille_go > 10:
        print("Au dessus du seuil gratuit BigQuery, reduire TOP_N_POPULAR et relancer.")
    else:
        print("Sous le seuil, fichier pret pour le notebook d'analyse.")

    con.register("controversy_games_df", pd.DataFrame({"game": CONTROVERSY_GAMES}))
    repartition = con.execute(f"""
        SELECT p.game, COUNT(*) AS n
        FROM read_parquet('{OUTPUT_PARQUET_PATH}') p
        JOIN controversy_games_df c ON p.game = c.game
        GROUP BY p.game
        ORDER BY n DESC
    """).fetchdf()
    print("\nRepartition des jeux a controverse dans le fichier final :")
    print(repartition)


if __name__ == "__main__":
    main()
