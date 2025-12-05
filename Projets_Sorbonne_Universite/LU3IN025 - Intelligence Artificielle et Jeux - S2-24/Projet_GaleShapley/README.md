# Projet d'Affectation Étudiants-Parcours

Ce projet implémente divers algorithmes et méthodes pour l'affectation des étudiants aux parcours en utilisant des préférences et des capacités. Voici une description détaillée des fonctionnalités et des fonctions disponibles dans le fichier `projet.py`.

## Fonctions Principales

### Lecture des Préférences

1. **`lire_pref_etudiants(fichier)`**
   - Lit les préférences des étudiants à partir d'un fichier.
   - **Paramètres**:
     - `fichier`: Chemin vers le fichier contenant les préférences des étudiants.
   - **Retourne**: Une liste de listes représentant les préférences des étudiants.

2. **`lire_pref_parcours(fichier)`**
   - Lit les préférences des parcours et leurs capacités à partir d'un fichier.
   - **Paramètres**:
     - `fichier`: Chemin vers le fichier contenant les préférences des parcours.
   - **Retourne**: Une liste de listes représentant les préférences des parcours et une liste des capacités.

### Algorithmes d'Affectation

3. **`gale_shapley_etudiants(CE, CP, capacites)`**
   - Implémente l'algorithme de Gale-Shapley côté étudiants pour l'affectation stable.
   - **Paramètres**:
     - `CE`: Préférences des étudiants.
     - `CP`: Préférences des parcours.
     - `capacites`: Capacités des parcours.
   - **Retourne**: Un dictionnaire représentant l'affectation des étudiants aux parcours.

4. **`gale_shapley_parcours(CE, CP, capacites)`**
   - Implémente l'algorithme de Gale-Shapley côté parcours pour l'affectation stable.
   - **Paramètres**:
     - `CE`: Préférences des étudiants.
     - `CP`: Préférences des parcours.
     - `capacites`: Capacités des parcours.
   - **Retourne**: Un dictionnaire représentant l'affectation des étudiants aux parcours.

### Vérification de la Stabilité

5. **`trouver_paires_instables(CE, CP, affectation)`**
   - Trouve les paires instables dans une affectation donnée.
   - **Paramètres**:
     - `CE`: Préférences des étudiants.
     - `CP`: Préférences des parcours.
     - `affectation`: Dictionnaire d'affectation des étudiants aux parcours.
   - **Retourne**: Une liste de tuples représentant les paires instables.

6. **`afficher_paires_instables(paires_instables)`**
   - Affiche les paires instables trouvées.
   - **Paramètres**:
     - `paires_instables`: Liste de tuples représentant les paires instables.

### Génération de Préférences Aléatoires

7. **`generer_pref_etudiants(n)`**
   - Génère des préférences aléatoires pour les étudiants.
   - **Paramètres**:
     - `n`: Nombre d'étudiants.
   - **Retourne**: Une liste de listes représentant les préférences des étudiants.

8. **`generer_pref_parcours(n)`**
   - Génère des préférences aléatoires pour les parcours.
   - **Paramètres**:
     - `n`: Nombre d'étudiants.
   - **Retourne**: Une liste de listes représentant les préférences des parcours.

9. **`generer_capacites(n, nb_parcours=9)`**
   - Génère des capacités équilibrées pour les parcours.
   - **Paramètres**:
     - `n`: Nombre d'étudiants.
     - `nb_parcours`: Nombre de parcours (par défaut 9).
   - **Retourne**: Une liste des capacités des parcours.

### Mesure du Temps d'Exécution

10. **`mesurer_temps_gale_shapley(n, nb_tests)`**
    - Mesure le temps moyen d'exécution des algorithmes de Gale-Shapley pour une valeur donnée de `n`.
    - **Paramètres**:
      - `n`: Nombre d'étudiants.
      - `nb_tests`: Nombre de tests à effectuer.
    - **Retourne**: Les temps moyens d'exécution pour les algorithmes côté étudiants et côté parcours.

11. **`tracer_courbes(nb_tests=100)`**
    - Trace les courbes de temps d'exécution des algorithmes de Gale-Shapley pour différentes valeurs de `n`.
    - **Paramètres**:
      - `nb_tests`: Nombre de tests à effectuer (par défaut 100).

### Génération de Fichiers LP

12. **`generate_lp_file_k(n, preferences, capacities, k, filename="problem.lp")`**
    - Génère un fichier LP pour s'assurer qu'un étudiant obtient un de ses `k` premiers choix.
    - **Paramètres**:
      - `n`: Nombre d'étudiants.
      - `preferences`: Préférences des étudiants.
      - `capacities`: Capacités des parcours.
      - `k`: Nombre maximum de choix considérés pour chaque étudiant.
      - `filename`: Nom du fichier LP généré.

13. **`generate_lp_file_maximum(n, m, preferences_students, preferences_parcours, capacities, k, filename="problem.lp")`**
    - Génère un fichier LP pour maximiser l'utilité totale des étudiants et des parcours.
    - **Paramètres**:
      - `n`: Nombre d'étudiants.
      - `m`: Nombre de parcours.
      - `preferences_students`: Préférences des étudiants.
      - `preferences_parcours`: Préférences des parcours.
      - `capacities`: Capacités des parcours.
      - `k`: Nombre maximum de choix de cours pour chaque étudiant.
      - `filename`: Nom du fichier LP généré.

### Optimisation de l'Utilité

14. **`find_min_k(n, preferences_students, capacities)`**
    - Trouve la valeur minimale de `k` telle que chaque étudiant puisse obtenir au moins un cours parmi ses `k` premiers choix.
    - **Paramètres**:
      - `n`: Nombre d'étudiants.
      - `preferences_students`: Préférences des étudiants.
      - `capacities`: Capacités des parcours.
    - **Retourne**: La valeur minimale de `k`.

15. **`maximize_min_utility(n, m, preferences_students, preferences_parcours, capacities, k)`**
    - Maximise l'utilité minimale des étudiants pour un `k` donné.
    - **Paramètres**:
      - `n`: Nombre d'étudiants.
      - `m`: Nombre de parcours.
      - `preferences_students`: Préférences des étudiants.
      - `preferences_parcours`: Préférences des parcours.
      - `capacities`: Capacités des parcours.
      - `k`: Nombre de premiers choix considérés pour chaque étudiant.
    - **Retourne**: Affectation optimale et utilité minimale atteinte.

16. **`maximize_utility_and_fairness(n, m, preferences_students, preferences_parcours, capacities, k)`**
    - Maximise la somme des utilités tout en garantissant une utilité minimale pour les étudiants.
    - **Paramètres**:
      - `n`: Nombre d'étudiants.
      - `m`: Nombre de parcours.
      - `preferences_students`: Préférences des étudiants.
      - `preferences_parcours`: Préférences des parcours.
      - `capacities`: Capacités des parcours.
      - `k`: Nombre de premiers choix considérés pour chaque étudiant.
    - **Retourne**: Affectation optimale, utilité minimale atteinte et utilité moyenne.

### Fonctions d'Affichage

17. **`afficher_matrice(titre, matrice, prefixe_ligne="")`**
    - Affiche une matrice avec un titre et un préfixe pour chaque ligne.
    - **Paramètres**:
      - `titre`: Titre de la matrice.
      - `matrice`: Matrice à afficher.
      - `prefixe_ligne`: Préfixe pour chaque ligne.

18. **`afficher_affectations_etudiants(titre, affectations)`**
    - Affiche les affectations des étudiants aux parcours.
    - **Paramètres**:
      - `titre`: Titre de l'affichage.
      - `affectations`: Dictionnaire d'affectation des étudiants aux parcours.

19. **`afficher_affectations_parcours(titre, affectations)`**
    - Affiche les affectations des parcours aux étudiants.
    - **Paramètres**:
      - `titre`: Titre de l'affichage.
      - `affectations`: Dictionnaire d'affectation des parcours aux étudiants.

### Fonction Principale

20. **`main()`**
    - Fonction principale qui gère le menu interactif pour exécuter les différentes fonctionnalités du projet.

## Utilisation

Pour utiliser ce projet, exécutez le fichier `projet.py`. Un menu interactif s'affichera, vous permettant de choisir parmi différentes options pour lire les préférences, exécuter les algorithmes d'affectation, générer des préférences aléatoires, tracer des courbes de temps d'exécution, générer des fichiers LP, et optimiser l'utilité des affectations.

## Dépendances

- `collections`
- `heapq`
- `random`
- `time`
- `matplotlib.pyplot`
- `numpy`
- `subprocess`
- `gurobipy`
- `pulp`

Assurez-vous d'avoir installé les bibliothèques nécessaires avant d'exécuter le projet. Vous pouvez les installer via pip si nécessaire :

```bash
pip install matplotlib numpy gurobipy pulp
```

## Remarques

Ce projet est conçu pour être utilisé dans un environnement éducatif ou de recherche pour étudier les algorithmes d'affectation et les méthodes d'optimisation. Les fichiers de préférences `PrefEtu.txt` et `PrefSpe.txt` doivent être présents dans le même répertoire que le script pour que les fonctions de lecture fonctionnent correctement.

## Auteurs

- [ZHANG Yuxiang] [LECOMTE Antoine]