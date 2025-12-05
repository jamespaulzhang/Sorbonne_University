# Projet d'Algorithmes de Couverture de Sommets (Vertex Cover)

Ce projet implémente une gamme complète d'algorithmes pour résoudre le problème de la couverture de sommets (Vertex Cover) dans les graphes non orientés, incluant des méthodes exactes, approximatives et heuristiques.

## 📋 Table des matières

- [Description du Problème](#description-du-problème)
- [Fonctionnalités](#fonctionnalités)
- [Algorithmes Implémentés](#algorithmes-implémentés)
- [Structure du Code](#structure-du-code)
- [Installation et Dépendances](#installation-et-dépendances)
- [Utilisation](#utilisation)
- [Format des Fichiers de Graphe](#format-des-fichiers-de-graphe)
- [Exemples d'Utilisation](#exemples-dutilisation)
- [Résultats et Performances](#résultats-et-performances)
- [Auteurs](#auteurs)

## 🎯 Description du Problème

Le problème de la couverture de sommets (Vertex Cover) consiste à trouver le plus petit ensemble de sommets dans un graphe tel que chaque arête soit incidente à au moins un sommet de cet ensemble. Ce problème est NP-complet et a de nombreuses applications pratiques en réseaux, bio-informatique et optimisation.

## ✨ Fonctionnalités

- **Lecture de graphes** depuis des fichiers texte
- **Génération de graphes aléatoires** selon le modèle G(n,p)
- **Algorithmes exacts** :
  - Force brute (pour petits graphes)
  - Algorithmes de branchement avec différentes stratégies
- **Algorithmes approximatifs** :
  - Algorithme de couplage (rapport 2)
  - Algorithme glouton
- **Heuristiques avancées** :
  - Recherche locale
  - Heuristique de degré pondéré
  - Heuristique hybride
- **Benchmark complet** avec comparaisons de performance
- **Visualisation** des résultats avec matplotlib
- **Interface utilisateur interactive** en ligne de commande

## 🧮 Algorithmes Implémentés

### Algorithmes Exactes
1. **Force Brute** - Exploration exhaustive (n ≤ 15)
2. **Branchement Simple** - Algorithme de base
3. **Branchement avec Bornes** - Utilise des bornes inférieures pour l'élagage
4. **Branchement Amélioré v1-v3** - Stratégies avancées de branchement

### Algorithmes Approximatifs
1. **Algorithme de Couplage** - Rapport d'approximation 2
2. **Algorithme Glouton** - Sélection par degré maximal

### Heuristiques
1. **Heuristique Aléatoire** - Sélection aléatoire pondérée
2. **Recherche Locale** - Amélioration itérative
3. **Heuristique Hybride** - Combinaison de méthodes

## 🏗️ Structure du Code

### Classes Principales

- **`Graph`** : Classe principale représentant un graphe
  - `__init__(adj)` : Initialisation avec liste d'adjacence
  - `sommets()`, `aretes()`, `degree()` : Opérations de base
  - `algo_couplage()`, `algo_glouton()` : Algorithmes approximatifs
  - `branchement_*()` : Famille d'algorithmes de branchement
  - `heuristique_*()` : Méthodes heuristiques

### Fonctions Utilitaires

- **`read_graph(filename)`** : Lecture depuis fichier
- **`generate_random_graph(n, p)`** : Génération aléatoire
- **`tests_algos()`** : Tests de performance comparatifs
- **`tracer_comparaison_*()`** : Fonctions de visualisation

## 🔧 Installation et Dépendances

### Prérequis
- Python 3.6+
- Bibliothèques requises :

```bash
pip install matplotlib
```

### Installation

1. Clonez ou téléchargez le fichier `vertex_cover.py`
2. Assurez-vous que les dépendances sont installées
3. Exécutez le programme :

```bash
python vertex_cover.py
```

## 🚀 Utilisation

### Mode Interactif

Le programme démarre en mode interactif avec un menu complet :

```python
# Exécuter le programme
python vertex_cover.py
```

### Menu Principal

1. **Charger un graphe** depuis un fichier
2. **Générer un graphe aléatoire**
3. **Afficher les informations** du graphe courant
4. **Tester les algorithmes** glouton et couplage
5. **Test interactif de branchement**
6. **Comparaison statistique** des algorithmes
7. **Tests de performance** (benchmark)
8. **Vérification par force brute**
9. **Test des branchements améliorés**
10. **Évaluation de la qualité** des approximations
11. **Évaluation des heuristiques**
12. **Tester tous les algorithmes**
13. **Quitter**

### Utilisation Programmatique

```python
from vertex_cover import Graph, read_graph, generate_random_graph

# Charger un graphe depuis un fichier
graphe = Graph(read_graph("mon_graphe.txt"))

# Ou générer un graphe aléatoire
graphe_aleatoire = generate_random_graph(20, 0.3)

# Utiliser différents algorithmes
couverture_couplage = graphe.algo_couplage()
couverture_glouton = graphe.algo_glouton()
couverture_optimale, noeuds = graphe.branchement_ameliore_v3()

# Vérifier la validité
est_valide = graphe.est_couverture_valide(couverture_couplage)
```

## 📁 Format des Fichiers de Graphe

Les graphes sont stockés dans des fichiers texte avec le format suivant :

```
Sommets
1
2
3
4
5
Nombre d aretes
Aretes
1 2
1 3
2 4
3 4
4 5
```

## 💡 Exemples d'Utilisation

### Exemple 1 : Test Rapide

```python
# Test avec un graphe chemin simple
G = Graph({0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]})

print("Couplage:", G.algo_couplage())
print("Glouton:", G.algo_glouton())
print("Branchement:", G.branchement_simple())
```

### Exemple 2 : Benchmark Complet

```python
# Tests de performance sur graphes aléatoires
results = tester_strategies_branchement(
    n_values=[8, 10, 12, 14, 16],
    p_values_labels=[0.1, 0.3, 0.5],
    num_instances=5
)

tracer_comparaison_strategies_complet(results)
```

### Exemple 3 : Évaluation de Qualité

```python
# Évaluer les rapports d'approximation
results = evaluer_qualite_approximation(
    n_values=[10, 15, 20],
    p_values=[0.2, 0.4],
    num_instances=10
)

tracer_rapports_approximation(results)
```

## 📊 Résultats et Performances

### Complexités

- **Force brute** : O(2^n)
- **Algorithmes de branchement** : O(1.47^n) à O(1.38^n) selon les améliorations
- **Algorithmes approximatifs** : O(m + n) pour couplage, O(n²) pour glouton

### Garanties d'Approximation

- **Algorithme de couplage** : Rapport 2
- **Algorithme glouton** : Rapport log(n) dans le pire cas

### Performances Pratiques

- **Graphes jusqu'à 15 sommets** : Solutions optimales rapides
- **Graphes 15-25 sommets** : Algorithmes de branchement efficaces
- **Graphes > 25 sommets** : Algorithmes approximatifs recommandés

## 👥 Auteurs

Ce projet a été développé dans le cadre d'un cours d'algorithmique avancée, implémentant et comparant diverses stratégies pour le problème de la couverture de sommets.

## 📄 Licence

Ce code est fourni à des fins éducatives et de recherche.