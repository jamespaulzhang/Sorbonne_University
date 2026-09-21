# Fantasy Cyclisme - Optimisation d'équipe

Application d'aide à la décision pour composer une équipe de fantasy cyclisme à partir des résultats du Tour de France. Le projet utilise un modèle de programmation linéaire en nombres entiers pour sélectionner les coureurs, gérer les abandons et analyser la robustesse des choix obtenus.

Ce travail a été réalisé dans le cadre du Master AI2D à Sorbonne Université.

## Objectif

Dans un jeu de fantasy cyclisme, il faut choisir une équipe de coureurs capable de marquer le plus de points possible tout en respectant un budget. Le problème devient plus difficile lorsque des coureurs abandonnent, que les transferts sont possibles seulement à certains moments et que les achats en cours de Tour sont plus chers.

L'objectif du projet est donc double :

- trouver une équipe optimale sous contraintes de budget, de taille d'équipe et de disponibilité des coureurs ;
- proposer des outils d'analyse pour comprendre pourquoi certains coureurs sont choisis, exclus ou proches de la solution optimale.

## Fonctionnalités

- Optimisation d'une équipe de fantasy cyclisme avec PuLP et le solveur CBC.
- Gestion de trois périodes de décision : début du Tour, après l'étape 10 et après l'étape 15.
- Prise en compte des abandons de coureurs.
- Gestion des achats et ventes lors des transferts.
- Surcoût de 10 % pour les achats effectués en période de transfert.
- Comparaison entre une stratégie avec transferts et une stratégie sans transfert.
- Affichage des équipes sélectionnées par période.
- Détail des transferts, budgets restants, points par étape et bonus finaux.
- Analyse de la dernière période avec ratios points/prix, points/coût de transfert et couches de Pareto.
- Analyse de sensibilité des points pour repérer les coureurs robustes ou proches d'entrer dans l'équipe.
- Simulation Monte Carlo, appelée "Facteur Chance", pour tester la stabilité des sélections lorsque les points, les coûts et la taille d'équipe varient.
- Export CSV des résultats d'analyse.

## Structure du dépôt

```text
.
|-- app_pulp.py           # Application Streamlit et modèle d'optimisation PuLP
|-- trait_donnees.py      # Lecture des données et calcul des points fantasy
|-- tdf_rankings/         # Données d'entrée du Tour de France
|-- doc/                  # Documents du projet
|-- anciennes_versions/   # Anciens scripts et prototypes conservés à part
|-- requirements.txt      # Dépendances Python à installer avec pip
|-- README.md             # Présentation et mode d'emploi du projet
`-- .python-version       # Version Python utilisée pour le projet
```

## Fichiers principaux

`app_pulp.py` contient l'application Streamlit. C'est le fichier à lancer pour utiliser l'interface. Il construit le modèle d'optimisation, résout le problème avec PuLP/CBC, affiche les résultats et lance les analyses post-optimales.

`trait_donnees.py` prépare les données utilisées par le modèle. Il lit les fichiers CSV du dossier `tdf_rankings/`, calcule les points obtenus par les coureurs sur les étapes, ajoute les bonus de maillots, la combativité et les points du classement final.

`tdf_rankings/` contient les données d'entrée :

- `valeurs.csv` : coût des coureurs ;
- `stage_01_2025.csv` à `stage_21_2025.csv` : résultats des étapes ;
- `maillots.csv` : bonus liés aux maillots et à la combativité ;
- `final.csv` : bonus du classement final.

`anciennes_versions/` regroupe les anciens scripts du projet. Ils sont gardés pour mémoire, mais l'application finale se lance depuis `app_pulp.py`.

## Installation

Le projet utilise Python 3.11.

Créer un environnement virtuel :

```powershell
python -m venv .venv
```

Activer l'environnement sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```powershell
pip install -r requirements.txt
```

Les dépendances principales sont :

- `streamlit` pour l'interface ;
- `pandas` pour la manipulation des données ;
- `pulp` pour l'optimisation linéaire entière ;
- `matplotlib` pour les graphiques.

## Lancement

Depuis la racine du projet :

```powershell
streamlit run app_pulp.py
```

L'application s'ouvre ensuite dans le navigateur.

## Utilisation

Dans la barre latérale, l'utilisateur peut régler :

- le budget initial ;
- la taille de l'équipe ;
- le nombre de scénarios pour la simulation Monte Carlo ;
- l'écart-type appliqué aux points ;
- l'écart-type appliqué aux coûts ;
- l'écart-type appliqué à la taille d'équipe ;
- la graine aléatoire de simulation.

Trois actions principales sont disponibles :

- `Analyse classique avec transferts` : résout le modèle avec transferts autorisés après les abandons ;
- `Analyse sans transfert` : résout le modèle en gardant la même équipe sur toute la course ;
- `Lancer les simulations` : génère plusieurs scénarios perturbés pour analyser la stabilité des choix.

Les résultats sont organisés en onglets :

- `Équipes` : composition retenue à chaque période ;
- `Transferts` : coureurs entrants, sortants et flux financiers ;
- `Points et scores` : détail des points par coureur et par étape ;
- `Histogrammes P15` : indicateurs points/prix, points/coût de transfert et couches de Pareto ;
- `Sensibilité points` : seuils de baisse ou de hausse nécessaires pour modifier la solution ;
- `Export analyse` : téléchargement des données d'analyse au format CSV.

## Modèle d'optimisation

Le problème est formulé comme un programme linéaire en nombres entiers.

Le modèle choisit les coureurs sélectionnés à chaque période :

- période 0 : étapes 1 à 10 ;
- période 10 : étapes 11 à 15 ;
- période 15 : étapes 16 à 21, avec ajout des bonus du classement final.

L'objectif est de maximiser le score total de l'équipe.

Les contraintes principales sont :

- respecter la taille d'équipe demandée à chaque période ;
- respecter le budget initial ;
- mettre à jour le budget après chaque achat et chaque vente ;
- empêcher la sélection d'un coureur après son abandon ;
- assurer la cohérence entre présence dans l'équipe, achat et vente ;
- interdire les transferts dans le mode "sans transfert".

Les principales variables de décision sont :

- `x[p][t]` : vaut 1 si le coureur `p` est sélectionné à la période `t` ;
- `achat[p][t]` : vaut 1 si le coureur `p` est acheté à la période `t` ;
- `vente[p][t]` : vaut 1 si le coureur `p` est vendu à la période `t` ;
- `budget[t]` : budget restant à la période `t`.

## Analyses post-optimales

L'analyse de sensibilité mesure la stabilité d'un coureur dans la solution. Pour un coureur sélectionné, elle estime la baisse de points nécessaire pour qu'il sorte de l'équipe. Pour un coureur non sélectionné, elle estime la hausse de points nécessaire pour qu'il devienne intéressant.

Les couches de Pareto permettent de comparer les coureurs de la dernière période selon deux critères : le coût et les points comptés. Un coureur en couche 1 n'est dominé par aucun autre coureur selon ces deux critères.

La simulation Monte Carlo perturbe les points, les coûts et la taille d'équipe. Elle permet de repérer les coureurs qui restent souvent sélectionnés malgré l'incertitude, ainsi que ceux qui deviennent intéressants seulement dans certains scénarios favorables.

## Remarque sur les données

Les fichiers CSV nécessaires au calcul sont fournis dans `tdf_rankings/`. Si le projet est déplacé sur une autre machine, il faut vérifier que `trait_donnees.py` pointe bien vers ce dossier de données.

## Contexte académique

Ce projet combine optimisation, analyse de données et interface interactive autour d'un cas d'usage concret en sport analytics. Il illustre comment un modèle de programmation linéaire peut être complété par des outils d'analyse pour transformer une solution optimale en véritable aide à la décision.
