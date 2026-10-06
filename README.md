# Classification de nœuds dans un réseau d'aéroports avec un GCN

## Présentation

Ce projet étudie une tâche de classification de nœuds sur un réseau d'aéroports.

Le graphe contient 3363 aéroports. Chaque nœud représente un aéroport et les arêtes représentent les connexions entre les aéroports.

L'objectif est de prédire le pays associé à chaque aéroport à l'aide d'un Graph Convolutional Network (GCN).

## Données

Les données sont fournies sous la forme d'un fichier GraphML.

Les informations utilisées pour chaque aéroport sont :

- `population` : population de la ville ;
- `lat` : latitude ;
- `lon` : longitude ;
- `country` : pays associé à l'aéroport ;
- `city_name` : nom de la ville.

Pour la classification, les entrées et la cible sont définies par :

```text
x = [population, latitude, longitude]
y = country
```

Le jeu de données contient 212 pays différents.

## Préparation des données

Le graphe est d'abord chargé avec NetworkX afin d'explorer sa structure et les attributs disponibles.

Les principales étapes de préparation sont :

1. vérification des valeurs manquantes ;
2. encodage de `country` avec `LabelEncoder` ;
3. construction des tenseurs contenant les caractéristiques et les labels ;
4. standardisation de `population`, `latitude` et `longitude` avec `StandardScaler` ;
5. conversion du graphe NetworkX vers PyTorch Geometric ;
6. création des ensembles d'entraînement, de validation et de test.

Le découpage utilisé contient :

| Ensemble | Nombre de nœuds |
|---|---:|
| Train | 2310 |
| Validation | 467 |
| Test | 586 |

Le découpage tient compte de la forte différence de représentation entre les pays afin de conserver les différentes classes dans l'ensemble d'entraînement.

## Modèle GCN

Le premier modèle étudié est un Graph Convolutional Network à deux couches.

```text
3 caractéristiques
       |
       v
   GCNConv
     32
       |
     ReLU
       |
       v
   GCNConv
     212
       |
       v
 pays prédit
```

La première couche construit une représentation des aéroports à partir de leurs caractéristiques et de leur voisinage dans le graphe.

La deuxième couche produit un score pour chacune des 212 classes.

L'entraînement utilise :

- Adam ;
- un learning rate de `0.01` ;
- `CrossEntropyLoss` ;
- les masques train/validation/test.

## Sélection du modèle

Un premier entraînement long de 10 000 epochs a donné :

| Train | Validation | Test |
|---:|---:|---:|
| 99,00 % | 75,16 % | 76,11 % |

La différence importante entre les performances d'entraînement et de validation indique un surapprentissage.

Une deuxième expérience conserve donc les paramètres correspondant à la meilleure accuracy de validation au lieu de sélectionner automatiquement la dernière epoch.

Le meilleur checkpoint obtenu donne :

| Train | Validation | Test |
|---:|---:|---:|
| 94,63 % | 79,66 % | **76,96 %** |

La meilleure validation a été obtenue autour de l'epoch 1205 lors de l'expérience conservée dans le notebook.

## Expérience avec dropout

Une architecture plus profonde avec dropout a également été testée afin de limiter le surapprentissage.

Elle utilise trois couches GCN et un dropout de 0.5 entre les couches.

Les résultats obtenus sont :

| Train | Validation | Test |
|---:|---:|---:|
| 81,04 % | 74,52 % | 72,35 % |

Le dropout réduit l'écart entre les performances d'entraînement et de test, mais diminue également l'accuracy finale dans cette configuration.

Le GCN simple avec sélection du meilleur checkpoint est donc conservé pour la suite de l'analyse.

## Baseline

Une baseline basée sur la classe majoritaire est utilisée comme point de comparaison.

Elle obtient une accuracy de :

```text
16,89 %
```

Le GCN atteint `76,96 %` sur le même ensemble de test.

## Analyse du déséquilibre des classes

Le nombre d'aéroports varie fortement selon les pays.

Les 212 classes ont été séparées en trois groupes :

| Groupe | Nombre de pays |
|---|---:|
| 1 à 5 aéroports | 123 |
| 6 à 20 aéroports | 49 |
| Plus de 20 aéroports | 40 |

Les performances du GCN varient fortement selon ces groupes :

| Groupe | Accuracy |
|---|---:|
| Pays rares | 10,87 % |
| Pays intermédiaires | 61,06 % |
| Pays fréquents | **88,29 %** |

Le déséquilibre des classes constitue donc une difficulté importante du problème.

## Analyse des prédictions

Sur les 586 aéroports de test :

```text
451 classifications correctes
135 classifications incorrectes
```

soit une accuracy de `76,96 %`.

Quelques prédictions correctes obtenues sont :

```text
Bora Bora       -> FRENCH_POLYNESIA
Honolulu        -> USA
Melbourne       -> AUSTRALIA
Hiroshima       -> JAPAN
New Plymouth    -> NEW_ZEALAND
```

Le modèle produit également certaines confusions :

```text
Auckland        : NEW_ZEALAND -> AUSTRALIA
Kuala Lumpur    : MALAYSIA -> THAILAND
Seoul           : SOUTH_KOREA -> CHINA
Toronto         : CANADA -> USA
Zurich          : SWITZERLAND -> THE_NETHERLANDS
```

Ces résultats montrent que l'accuracy globale masque des différences importantes entre les classes, notamment pour les pays disposant de très peu d'aéroports.

## Résultats actuels

| Méthode | Train | Validation | Test |
|---|---:|---:|---:|
| Classe majoritaire | - | - | 16,89 % |
| GCN - 10 000 epochs | 99,00 % | 75,16 % | 76,11 % |
| GCN - meilleur checkpoint | 94,63 % | **79,66 %** | **76,96 %** |
| GCN avec dropout | 81,04 % | 74,52 % | 72,35 % |

Ces résultats correspondent à l'état actuel du projet. D'autres méthodes de comparaison pourront être ajoutées dans la suite du travail.

## Reproductibilité

### Environnement

Le projet utilise principalement :

- Python ;
- PyTorch ;
- PyTorch Geometric ;
- NetworkX ;
- scikit-learn ;
- NumPy ;
- Matplotlib.

### Installation

Les dépendances nécessaires peuvent être installées avec :

```bash
pip install torch torch-geometric networkx scikit-learn numpy matplotlib
```

### Exécution

Placer le fichier GraphML dans le même dossier que le notebook :

```text
airportsAndCoordAndPop.graphml
```

Puis exécuter les cellules du notebook dans l'ordre :

```text
PROJET_GNN1.ipynb
```

Le notebook effectue successivement :

```text
chargement du graphe
        ↓
exploration
        ↓
préparation des features et labels
        ↓
normalisation
        ↓
conversion PyTorch Geometric
        ↓
création des masks
        ↓
entraînement du GCN
        ↓
sélection du meilleur checkpoint
        ↓
évaluation
        ↓
analyse des erreurs
```

Une graine aléatoire est utilisée dans certaines expériences afin de faciliter la reproductibilité des résultats.

## Utilisation d'une IA générative

ChatGPT, avec le modèle **GPT-5.6 Sol**, a été utilisé comme outil d'assistance pendant le développement du projet.

Son utilisation a principalement concerné :

- l'explication de concepts liés aux GNN et à PyTorch Geometric ;
- l'aide à la compréhension et à la structuration du code ;
- l'analyse des résultats obtenus ;
- l'amélioration de la présentation et de la documentation du notebook.

Les expériences ont été exécutées dans le notebook et les résultats présentés dans ce projet proviennent de ces exécutions.

Exemples de demandes utilisées pendant le travail :

```text
"Explique-moi comment préparer x et y pour une classification
de nœuds avec PyTorch Geometric."

"Comment créer les masques train, validation et test ?"

"Explique-moi la différence entre train(), eval()
et torch.no_grad()."

"Comment interpréter l'écart entre l'accuracy train
et l'accuracy test de mon GCN ?"

"Comment analyser les performances selon le nombre
d'aéroports disponibles pour chaque pays ?"
```

## État du projet

Le travail présenté ici correspond à la première phase du projet :

- préparation du réseau d'aéroports ;
- mise en place de la tâche de classification ;
- entraînement d'un GCN ;
- étude du surapprentissage ;
- sélection du meilleur checkpoint ;
- test d'une régularisation par dropout ;
- analyse des prédictions ;
- étude de l'influence du déséquilibre des classes.

La comparaison avec d'autres modèles et l'étude d'ablation seront ajoutées dans la suite du projet.
