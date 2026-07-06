# Détection de review bombing sur Steam Reviews

Réalisé par Rostom SAMAR — Master 1 Expert en Intelligence Artificielle

Projet de data science couvrant une chaîne complète : collecte, stockage, ETL, analyse exploratoire, machine learning (supervisé et non supervisé) et interface de démonstration, appliquée à la détection d'épisodes de review bombing dans les avis Steam.

## Table des matières

1. [Problématique](#1-problématique)
2. [Dataset](#2-dataset)
3. [Stockage](#3-stockage)
4. [Pipeline ETL](#4-pipeline-etl)
5. [Analyse exploratoire](#5-analyse-exploratoire-eda)
6. [Machine learning](#6-machine-learning)
7. [Évaluation et choix du modèle](#7-évaluation-et-choix-du-modèle)
8. [Interface de démonstration](#8-interface-de-démonstration)
9. [Difficultés rencontrées](#9-difficultés-rencontrées)
10. [Structure du dépôt](#10-structure-du-dépôt)
11. [Reproduire le projet](#11-reproduire-le-projet)

---

## 1. Problématique

Les avis Steam n'existant qu'après l'achat du jeu, la problématique retenue n'est pas de prédire le succès d'un jeu avant sa sortie, mais de **détecter les épisodes de review bombing** : des vagues d'avis négatifs concentrées dans le temps, déclenchées par un événement externe (changement d'économie de jeu, décision d'éditeur, controverse), et déconnectées de la qualité réelle du jeu.

Hypothèse centrale testée tout au long du projet : le texte d'un avis suffit-il, à lui seul, à détecter un tel épisode, sans recourir aux statistiques de volume ou de négativité ?

## 2. Dataset

| Caractéristique | Valeur |
|---|---|
| Source | [Steam Reviews — Kaggle](https://www.kaggle.com/datasets/kieranpoc/steam-reviews) |
| Volume brut | 113,9 millions de lignes, 39,6 Go décompressés |
| Structure | avis (texte, note, votes), métadonnées joueur, jeu, timestamps |
| Volume filtré | 32,3 millions d'avis, 506 jeux (28 à controverse documentée + 500 les plus reviewés) |

**Défis d'exploitation :** volume trop important pour un traitement cloud gratuit fiable (voir [difficultés rencontrées](#9-difficultés-rencontrées)) ; une première estimation du volume, obtenue via une session Colab instable, s'est révélée incomplète (téléchargement tronqué par une coupure réseau), corrigée par un téléchargement local fiable donnant le volume réel ci-dessus ; couverture partielle (tous les jeux Steam n'y figurent pas, par exemple Helldivers 2 est absent).

**Stratégie de filtrage :** sélection combinant des jeux à controverse documentée (War Thunder, Cyberpunk 2077, Team Fortress 2, GTA V, etc.) et les jeux les plus reviewés, plutôt qu'un échantillonnage aléatoire — un tirage aléatoire aurait dilué les épisodes rares qu'on cherche justement à détecter.

## 3. Stockage

**Choix : SQL (DuckDB)**, plutôt que NoSQL. Justification : données tabulaires à schéma fixe, analyse fondée sur des agrégations (comptages, corrélations), pas de besoin de flexibilité de schéma. DuckDB permet en plus un traitement 100 % local, sans dépendance cloud ni quota.

| Optimisation | Résultat mesuré |
|---|---|
| Indexation (`game`, `timestamp_created`, `voted_up`) | 0,828 s → 0,044 s (≈19x) |
| Partitionnement par jeu | 0,084 s sur la même requête |

## 4. Pipeline ETL

- **Extraction** : filtrage du CSV brut par jeu et langue anglaise, en un seul passage streamé (DuckDB), sans jamais charger le fichier complet en mémoire.
- **Transformation** : suppression des doublons (136 866) et des avis sans texte (12 334), conversion des timestamps, vérification de l'encodage binaire, normalisation z-score de 5 variables numériques.
- **Chargement** : table `reviews_clean` en base DuckDB (32 176 434 lignes), export Parquet pour les notebooks d'analyse.
- **Automatisation** : l'ensemble du pipeline (`scripts/01` à `06`) s'exécute en une série de scripts Python autonomes, sans intervention manuelle.

## 5. Analyse exploratoire (EDA)

- 88,36 % d'avis positifs sur l'ensemble du dataset filtré.
- Corrélations entre métadonnées numériques et recommandation toutes proches de zéro (`|r| < 0.1`) : le temps de jeu, la longueur de l'avis ou l'historique de l'auteur ne prédisent pas la recommandation.
- **Détection d'anomalies** (z-score de volume quotidien par jeu) : War Thunder (17-19 mai 2023) combine une anomalie de volume extrême (z = 29,2) et un taux d'avis négatifs de 88 % (contre 26 % en moyenne) — review bombing confirmé. Cyberpunk 2077 (décembre 2020), malgré un volume spectaculaire, ne présente qu'un écart de négativité de 5,7 points — un pic d'attention au lancement, pas un review bombing.

Conclusion clé : **un pic de volume seul ne suffit pas** à identifier un review bombing ; il faut le croiser avec un écart de négativité.

## 6. Machine learning

### Apprentissage supervisé

Étiquette dérivée de la règle validée en EDA (z-score de volume > 3 et écart de taux négatif > 0,20), appliquée aux 506 jeux — 270 034 avis positifs, équilibrés à 1 pour 5 avec les négatifs.

| Modèle | Précision | Recall | F1 (review bombing) |
|---|---|---|---|
| Régression logistique (TF-IDF) | 0,50 | 0,75 | **0,600** |
| Naive Bayes complémentaire | 0,47 | 0,76 | 0,577 |
| Régression logistique + LSA (100 comp.) | 0,41 | 0,68 | 0,509 |

La LSA dégrade la performance : le signal utile tient à des termes lexicaux précis, pas à une structure sémantique large. Une tentative d'embeddings de phrase (sentence-transformers) a été abandonnée suite à un bug confirmé, non résolu, dans les versions récentes de la librairie `transformers` (voir [difficultés rencontrées](#9-difficultés-rencontrées)).

### Apprentissage non supervisé

Clustering k-means (TF-IDF + LSA 50 composantes) sur l'ensemble du corpus, k choisi par score de silhouette (k=2 retenu). Résultat : un cluster majoritaire (95,2 %) proche de la moyenne générale de review bombing, et un cluster résiduel (4,8 %) où le review bombing est au contraire **sous-représenté**. Les termes représentatifs des deux clusters sont quasi identiques.

Résultat négatif mais informatif : la structure géométrique globale du texte ne suffit pas à isoler le review bombing sans étiquette — confirme que sa détection repose sur un signal supervisé fin, pas sur une thématique visible à l'œil nu.

## 7. Évaluation et choix du modèle

**Modèle retenu : régression logistique sur TF-IDF brut** (F1 = 0,600), pour l'interface de démonstration. Recall privilégié sur la précision : manquer un épisode de review bombing coûte plus cher qu'une fausse alerte.

## 8. Interface de démonstration

Application Streamlit (`app.py`), trois onglets :

- **Visualisation des données** : choix d'un jeu, courbe du volume quotidien d'avis avec surimpression des anomalies (même règle z-score > 3 que l'EDA, recalculée en direct via DuckDB), indicateurs de volume total et de taux d'avis négatifs.
- **Prédiction** : saisie libre d'un avis ou sélection d'un exemple réel tiré du jeu de données étiqueté, prédiction par le modèle retenu (régression logistique sur TF-IDF) avec la probabilité et les termes ayant le plus influencé la décision.
- **Méthodologie** : rappel des définitions et limites de l'étiquetage et du modèle.

**Lancement :**

```
pip install streamlit
streamlit run app.py
```

Nécessite au préalable que le pipeline complet ait été exécuté (`scripts/01` à `06`, puis `07_ml_supervised.ipynb` pour générer les modèles dans `models/`).

## 9. Difficultés rencontrées

- **Instabilité Google Colab** : plantages RAM, coupures réseau causant des téléchargements tronqués, sessions non persistantes — a motivé le passage à un traitement 100 % local avec DuckDB.
- **Téléchargement initial incomplet** : la première estimation du volume (obtenue via Colab, environ 21 Go) s'est révélée tronquée par une coupure réseau ; un téléchargement local fiable a donné le volume réel (39,6 Go, 113,9 millions de lignes), nécessitant de revalider la liste des jeux ciblés.
- **Environnement Python local** : décalage entre l'interpréteur du terminal et celui du noyau Jupyter, scan antivirus ralentissant l'accès aux fichiers volumineux.
- **Conflits de dépendances** (`sentence-transformers` / `transformers` / `torch`) : bug confirmé et documenté publiquement (renommage interne incohérent d'une classe centrale de `transformers`, introduit en octobre 2025), non résolu malgré plusieurs tentatives — a conduit à remplacer les embeddings par une méthode LSA, sans dépendance instable.

## 10. Structure du dépôt

```
scripts/                  Pipeline de données (filtrage, base, ETL, étiquetage)
01_count_games.py
02_filter_export.py
03_build_database.py
04_etl_pipeline.py
06_build_labels.py
05_eda.ipynb               Analyse exploratoire
07_ml_supervised.ipynb     Apprentissage supervisé et évaluation
08_clustering.ipynb        Apprentissage non supervisé
data/                      Données brutes et transformées (non versionnées)
models/                    Modèles entraînés (vectoriseur, régression logistique, Naive Bayes)
outputs/                   Figures générées (eda/, ml/, clustering/)
```

## 11. Reproduire le projet

```
pip install duckdb pandas matplotlib scikit-learn jupyter pyarrow

python scripts/01_count_games.py
python scripts/02_filter_export.py
python scripts/03_build_database.py
python scripts/04_etl_pipeline.py
python scripts/06_build_labels.py
```

Puis exécuter dans l'ordre `05_eda.ipynb`, `07_ml_supervised.ipynb`, `08_clustering.ipynb`.
