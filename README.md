# Classification de nœuds dans un réseau d'aéroports avec des GNN

## Présentation

Ce projet étudie une tâche de classification de nœuds sur un réseau d'aéroports.

Le graphe contient 3363 aéroports. Chaque nœud représente un aéroport et les arêtes représentent les connexions entre les aéroports.

L'objectif est de prédire le pays associé à chaque aéroport et de comparer plusieurs architectures : GCN, GAT, GraphSAGE et un MLP utilisé comme baseline.

## Données

Les données sont fournies sous la forme d'un fichier GraphML.

Les informations utilisées pour chaque aéroport sont :

- `population` : population de la ville ;
- `lat` : latitude ;
- `lon` : longitude ;
- `country` : pays associé à l'aéroport ;
- `city_name` : nom de la ville.

Pour la classification, les entrées et la cible sont initialement définies par :

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

Les mêmes ensembles train, validation et test sont utilisés pour comparer les différents modèles.

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

## Sélection du modèle GCN

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

## Baseline classe majoritaire

Une baseline basée sur la classe majoritaire est utilisée comme premier point de comparaison.

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

## Analyse des prédictions du GCN

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

## Modèles supplémentaires

### Graph Attention Network

Un Graph Attention Network (GAT) est ensuite testé afin d'introduire un mécanisme d'attention entre les aéroports voisins.

L'architecture utilisée comporte deux couches `GATConv`.

La première couche utilise deux têtes d'attention de dimension 32. Les sorties des deux têtes sont concaténées, ce qui produit une représentation de dimension 64.

La deuxième couche produit les scores correspondant aux 212 pays.

Le meilleur checkpoint du GAT avec les trois caractéristiques initiales donne :

| Train | Validation | Test |
|---:|---:|---:|
| 90,48 % | 81,37 % | 79,69 % |

### GraphSAGE

GraphSAGE est également testé avec deux couches `SAGEConv`.

La première couche transforme les trois caractéristiques en une représentation de dimension 32, suivie d'une fonction ReLU. La deuxième couche produit les scores des 212 classes.

L'agrégation utilisée est une agrégation par moyenne.

Les résultats obtenus sont :

| Train | Validation | Test |
|---:|---:|---:|
| 99,52 % | 80,30 % | 76,45 % |

### MLP

Un MLP est utilisé comme baseline ne prenant pas en compte la structure du graphe.

Il utilise uniquement les caractéristiques des aéroports et n'utilise donc pas `edge_index`.

L'architecture contient deux couches linéaires :

```text
3 caractéristiques
       |
       v
 Linear
   32
       |
     ReLU
       |
       v
 Linear
   212
       |
       v
 pays prédit
```

Les résultats obtenus sont :

| Train | Validation | Test |
|---:|---:|---:|
| 95,41 % | 85,22 % | 81,40 % |

Le bon résultat du MLP montre que les caractéristiques des aéroports contiennent déjà une grande quantité d'information permettant de prédire leur pays.

## Comparaison des modèles

Les principaux modèles sont comparés avec les mêmes ensembles d'entraînement, de validation et de test.

| Méthode | Train | Validation | Test |
|---|---:|---:|---:|
| Classe majoritaire | - | - | 16,89 % |
| GCN | 94,63 % | 79,66 % | 76,96 % |
| GraphSAGE | 99,52 % | 80,30 % | 76,45 % |
| GAT | 90,48 % | 81,37 % | 79,69 % |
| MLP | 95,41 % | **85,22 %** | **81,40 %** |

Sur cette première comparaison, le MLP obtient la meilleure accuracy test avec `81,40 %`.

Ces résultats suggèrent que les caractéristiques des nœuds, et notamment les informations géographiques, jouent un rôle très important dans cette tâche.

## Étude d'ablation du GAT

Une étude d'ablation est réalisée sur le GAT afin d'évaluer l'importance des différentes caractéristiques d'entrée.

Trois configurations sont comparées :

1. population + latitude + longitude ;
2. latitude + longitude ;
3. population uniquement.

Les résultats sont :

| Caractéristiques du GAT | Train | Validation | Test |
|---|---:|---:|---:|
| Population + latitude + longitude | 90,48 % | 81,37 % | 79,69 % |
| Latitude + longitude | 88,18 % | **84,15 %** | **81,57 %** |
| Population uniquement | 51,21 % | 48,82 % | 47,61 % |

La suppression de la population améliore l'accuracy test de `79,69 %` à `81,57 %`.

À l'inverse, lorsque seule la population est utilisée, l'accuracy chute à `47,61 %`.

Les coordonnées géographiques sont donc les caractéristiques les plus importantes pour cette tâche.

La configuration GAT utilisant uniquement la latitude et la longitude est retenue comme meilleure configuration GNN.

## Stabilité sur plusieurs seeds

Les deux meilleures configurations sont ensuite réentraînées avec cinq seeds différentes.

Les mêmes masques train, validation et test sont conservés afin que seule l'initialisation des poids varie.

Les seeds utilisées sont :

```text
0, 1, 2, 3, 4
```

### MLP

Les accuracies test obtenues sont :

```text
Seed 0 : 77,99 %
Seed 1 : 81,57 %
Seed 2 : 81,74 %
Seed 3 : 79,69 %
Seed 4 : 82,25 %
```

La moyenne obtenue est :

```text
80,65 % ± 1,59 %
```

### GAT avec latitude et longitude

Les accuracies test obtenues sont :

```text
Seed 0 : 81,23 %
Seed 1 : 83,79 %
Seed 2 : 80,72 %
Seed 3 : 83,11 %
Seed 4 : 83,28 %
```

La moyenne obtenue est :

```text
82,42 % ± 1,22 %
```

La configuration finale retenue est donc le **GAT utilisant uniquement la latitude et la longitude**, qui obtient la meilleure performance moyenne sur les cinq seeds.

## Reproductibilité

### Environnement

Le projet utilise principalement :

- Python ;
- PyTorch ;
- PyTorch Geometric ;
- NetworkX ;
- scikit-learn ;
- NumPy ;
- pandas ;
- Matplotlib.

### Installation

Les dépendances nécessaires peuvent être installées avec :

```bash
pip install torch torch-geometric networkx scikit-learn numpy pandas matplotlib
```

### Fichiers nécessaires

Le fichier du graphe doit être placé dans le même dossier que le notebook :

```text
airportsAndCoordAndPop.graphml
```

Le notebook principal est :

```text
PROJET_GNN.ipynb
```

### Exécution

Exécuter les cellules de `PROJET_GNN.ipynb` dans l'ordre.

Le notebook effectue successivement :

```text
chargement du graphe
        ↓
exploration des données
        ↓
préparation des features et labels
        ↓
standardisation
        ↓
conversion vers PyTorch Geometric
        ↓
création des masks train / validation / test
        ↓
entraînement et évaluation du GCN
        ↓
analyse du surapprentissage et dropout
        ↓
analyse du déséquilibre des classes
        ↓
entraînement du GAT
        ↓
entraînement du MLP
        ↓
entraînement de GraphSAGE
        ↓
comparaison des modèles
        ↓
étude d'ablation du GAT
        ↓
évaluation des meilleures configurations sur plusieurs seeds
        ↓
sélection du modèle final
```

Les mêmes masks train, validation et test sont conservés lors de la comparaison des différents modèles.

Pour l'étude de stabilité, les modèles sont réentraînés avec les seeds `0`, `1`, `2`, `3` et `4`.

Pour chaque entraînement concerné, le checkpoint présentant la meilleure accuracy de validation est conservé avant l'évaluation finale sur l'ensemble de test.

## Utilisation d'une IA générative

ChatGPT, avec le modèle **GPT-5.6 Sol**, a été utilisé comme outil d'assistance pendant le développement du projet.

Son utilisation a principalement concerné :

- l'explication de concepts liés aux GNN et à PyTorch Geometric ;
- l'aide à la compréhension et à la structuration du code ;
- l'aide au débogage de certaines expériences ;
- l'analyse et l'interprétation des résultats obtenus ;
- l'organisation des expériences de comparaison ;
- l'aide à la mise en place de l'étude d'ablation ;
- l'amélioration de la présentation et de la documentation du notebook et du projet.

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

"Comment entraîner un GraphSAGE avec les mêmes masks
que mon GCN ?"

"Comment réaliser une étude d'ablation sur les features
de mon GAT ?"

"Comment comparer la stabilité du MLP et du GAT
sur plusieurs seeds ?"
```

## État final du projet

Le projet comprend :

- la préparation et l'exploration du réseau d'aéroports ;
- la mise en place d'une tâche de classification de nœuds ;
- l'entraînement et l'évaluation d'un GCN ;
- l'étude du surapprentissage et d'une régularisation par dropout ;
- l'analyse du déséquilibre des classes ;
- l'analyse de certaines prédictions ;
- la comparaison avec un GAT, GraphSAGE et un MLP ;
- une étude d'ablation des caractéristiques utilisées par le GAT ;
- une évaluation de la stabilité des meilleures configurations sur plusieurs seeds.

La configuration finale retenue est un **GAT utilisant la latitude et la longitude**, avec une accuracy test moyenne de **82,42 % ± 1,22 %** sur cinq seeds.