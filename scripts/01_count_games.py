"""
Comptage des avis par jeu (langue anglaise uniquement), en local avec DuckDB.

Un seul passage sur le CSV brut, sans jamais le charger entierement en memoire.
Limite de RAM fixee explicitement pour eviter de saturer la machine.
"""

import duckdb

CSV_PATH = "data/steam-reviews/all_reviews/all_reviews.csv"
OUTPUT_COUNTS_PATH = "scripts/game_counts.csv"
MEMORY_LIMIT = "4GB"

KEYWORDS = [
    "war thunder",
    "grand theft auto",
    "crusader kings",
    "counter-strike",
    "dota",
    "terraria",
    "cyberpunk 2077",
    "elden ring",
    "apex legends",
    "destiny",
    "baldur's gate 3",
    "fall guys",
    "red dead redemption",
    "new world",
    "halo infinite",
    "hades",
    "helldivers",
    "team fortress",
    "pubg",
    "playerunknown",
    "kerbal space program",
    "assassin's creed unity",
    "sonic mania",
    "firewatch",
    "titan souls",
    "stardew",
]


def main():
    con = duckdb.connect(config={"memory_limit": MEMORY_LIMIT})

    total_rows = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{CSV_PATH}', ignore_errors=true)"
    ).fetchone()[0]
    print("Lignes lues au total :", total_rows)
    if total_rows < 50_000_000:
        print("Nombre de lignes plus bas qu'attendu (reference precedente : environ 58 millions). Verifier la taille du CSV avant de continuer.")

    counts = con.execute(f"""
        SELECT game, COUNT(*) AS review_count
        FROM read_csv_auto('{CSV_PATH}', ignore_errors=true)
        WHERE language = 'english'
        GROUP BY game
        ORDER BY review_count DESC
    """).fetchdf()

    print("Jeux distincts (avis en anglais) :", len(counts))
    print(counts.head(20))

    counts.to_csv(OUTPUT_COUNTS_PATH, index=False)
    print("Comptage complet sauvegarde dans", OUTPUT_COUNTS_PATH)

    print("\nRecherche des jeux a controverse par mot cle :")
    game_names = [name for name in counts["game"].tolist() if isinstance(name, str)]
    nb_noms_ignores = len(counts) - len(game_names)
    if nb_noms_ignores:
        print(f"{nb_noms_ignores} valeur(s) de jeu manquante(s) ou non textuelle(s) ignoree(s) dans la recherche.")

    for keyword in KEYWORDS:
        found = [name for name in game_names if keyword in name.lower()]
        print(keyword, "->", found)


if __name__ == "__main__":
    main()
