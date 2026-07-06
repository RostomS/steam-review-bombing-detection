# Mise en place locale

Traitement entierement local avec DuckDB : lecture directe du CSV sans le charger en memoire, un seul passage sur le fichier, sans dependance a une session cloud.

## 1. Installer les dependances

```
pip install duckdb pandas matplotlib scikit-learn jupyter pyarrow streamlit
```

## 2. Recuperer le dataset

Telecharger `steam-reviews.zip` depuis [kaggle.com/datasets/kieranpoc/steam-reviews](https://www.kaggle.com/datasets/kieranpoc/steam-reviews) et le decompresser dans `data/`, de maniere a obtenir :

```
data/steam-reviews/all_reviews/all_reviews.csv
```

## 3. Lancer le pipeline dans l'ordre

```
python scripts/01_count_games.py
python scripts/02_filter_export.py
python scripts/03_build_database.py
python scripts/04_etl_pipeline.py
python scripts/06_build_labels.py
```

Puis executer `05_eda.ipynb`, `07_ml_supervised.ipynb`, `08_clustering.ipynb`, dans cet ordre.

Detail complet de chaque etape, des resultats et des choix methodologiques : voir `README.md` a la racine du projet.
