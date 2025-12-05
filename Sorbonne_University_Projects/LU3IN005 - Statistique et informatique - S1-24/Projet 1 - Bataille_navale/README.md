# README - Jeu de Bataille Navale et Recherche d'Objets

## Auteurs
**Sam ASLO (ID: 21210657)**  
**Yuxiang ZHANG (ID: 21202829)**

## Description

Ce projet simule un jeu de bataille navale où des bateaux sont placés sur une grille de manière aléatoire. Le programme permet de vérifier si un bateau peut être placé à une position donnée, de compter le nombre total de configurations valides pour placer les bateaux, et d'afficher la grille avec les bateaux. Il permet aussi de modéliser un jeu entre deux joueurs machines qui mettent en œuvre des stratégies différentes. De plus, il inclut une fonctionnalité de recherche d'objets basée sur des distributions de probabilité.

## Structure du Projet

Le projet comprend plusieurs fichiers Python :

- **grille.py** : Définit la classe `Grille`, qui représente la grille de jeu.
- **bateau.py** : Définit la classe `Bateau`, qui représente les différents types de bateaux.
- **functions.py** : Définit les fonctions principales dans la partie 1 et 2.
- **bataille.py** : Définit la classe `Bataille`, qui représente la bataille navale entre deux joueurs.
- **joueur.py** : Définit la classe `Joueur`, qui représente un joueur avec différentes méthodes pour jouer.
- **main_partie1et2.py** : Contient la logique principale du programme pour la partie 1 et 2.
- **main_partie3.py** : Contient la logique principale du programme pour la partie 3.
- **bayesian_search_2d.py** : Contient les fonctions principales et les tests sur différentes distributions de la partie 4.

## Fonctions Principales

### Placement des Bateaux (pour la partie 1 et 2)

1. **peut_placer(grille, bateau, position, direction)** : Vérifie si un bateau peut être placé à une position donnée sur la grille dans une direction spécifique (horizontale ou verticale).
2. **place(grille, bateau, position, direction)** : Place un bateau sur la grille à une position et direction spécifiées.
3. **place_alea(grille, bateau)** : Place un bateau aléatoirement sur la grille après avoir vérifié les conflits.
4. **affiche(grille, destroy)** : Affiche la grille à l'aide de `matplotlib`.
5. **grilles_eq(grilleA, grilleB)** : Compare deux grilles pour vérifier leur égalité.
6. **genere_grille()** : Génère une grille avec des bateaux placés aléatoirement.
7. **nb_placer(grille, bateau)** : Calcule le nombre de façons possibles de placer un bateau sur la grille.
8. **nb_total_placer(grille, bateaux)** : Calcule le nombre total de façons possibles de placer une liste de bateaux sur la grille.
9. **nb_grilles(grille)** : Calcule combien de tentatives sont nécessaires pour générer une grille identique à une grille de référence.
10. **remove(grille, bateau)** : Retire un bateau de la grille à partir de sa position et direction actuelle.
11. **count_configs(grille, bateaux)** : Compte le nombre total de configurations valides pour placer une liste de bateaux sur la grille.

### Instructions pour `main_partie1et2.py`
- **Lancer le Programme** : Exécutez le script principal pour accéder au menu du jeu.
- **Choisir une Option** : Entrez le numéro correspondant à l'option que vous souhaitez utiliser.
- **Suivre les Instructions** : Suivez les instructions affichées pour entrer les coordonnées des bateaux, choisir un bateau, ou définir la taille de la grille.
- **Quitter le Programme** : Entrez `0` pour quitter le jeu.

### Modélisation Probabiliste du Jeu (pour la partie 3)

Pour cette partie, les trois fichiers importants sont `bataille.py`, `joueur.py` et `main_partie3.py`. Leurs fonctions principales sont :

#### `joueur.py`
- **jouer_un_coup** : Différente pour les trois types de joueurs, cette fonction permet de déterminer une position à attaquer.
- **enregistrer_resultat** : Étudie les résultats d'un tir pour déterminer s'il touche un bateau ou non. Si un bateau est touché, elle génère des positions cibles pour les attaques futures, en tenant compte du tir précédent. Utilisée par le JoueurHeuristique et le JoueurProbabiliste.
- **_clc_bat_proba** : Calcule les probabilités d'attaque pour une taille de bateau donnée. La probabilité pour chaque cas est stockée dans une matrice.

#### `bataille.py`
- **reset** : Réinitialise les grilles des deux joueurs et replace les bateaux aléatoirement.
- **match** : Permet de faire un match entre joueurs et appelle toutes les fonctions de la classe Joueur.

#### `main_partie3.py`
- **comparer_strategies** : Simule plusieurs parties de bataille et enregistre le nombre de coups nécessaires pour chaque partie.
- **main** : Permet de tester des jeux entre joueurs et génère les graphes.

### Recherche d'Objet (pour la partie 4)

- **initialize_grid(N, distribution_type)** : Initialise la grille avec une distribution de probabilité spécifiée (uniforme, coins, centre, bords, aléatoire).
- **detect_object(probability_grid, detection_success_rate)** : Simule la recherche d'un objet dans la grille en fonction de la distribution de probabilité et du taux de succès de détection.
- **update_probabilities(probability_grid, detected, i, j, detection_success_rate)** : Met à jour la grille de probabilité après une tentative de détection.
- **search_object(N, detection_success_rate, distribution_type, max_iterations)** : Effectue le processus de recherche d'un objet dans la grille avec mise à jour des probabilités.
- **plot_probability_distribution(probability_grid, title)** : Affiche la distribution de probabilité sur un graphique.

## Utilisation

Pour exécuter le programme, utilisez la commande suivante :

- **Pour la partie 1 et 2** :
  python3 main_partie1et2.py
  Cela générera une grille avec des bateaux placés aléatoirement et affichera des informations sur la possibilité de placer un bateau, ainsi que le nombre total de configurations valides.

- **Pour la partie 3** :
  python3 main_partie3.py
  Cela permet de tester trois batailles : une entre deux JoueurAleatoire, une autre entre deux JoueurHeuristique, et la troisième entre deux JoueurProbabiliste. Chaque bataille est répétée `nb_bataille` (100) fois avec des positions de bateaux différentes à chaque fois. Ensuite, les résultats sont tracés sous forme de graphique.

- **Pour la partie 4** :
  python3 bayesian_search_2d.py
  Le programme testera différentes distributions de probabilité pour la recherche d'objets et affichera les résultats.
