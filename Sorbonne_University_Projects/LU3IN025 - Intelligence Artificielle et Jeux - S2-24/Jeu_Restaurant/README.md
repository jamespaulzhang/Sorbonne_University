# restaurant-game
Template et instructions pour le projet 2025 IA et Jeux

## Présentation générale du projet

On propose dans ce projet d'implémenter un jeu stratégique reprenant certaines des notions vues en cours. 

### Phase de jeu principale
Chaque jour, plusieurs joueurs (8 par défaut) disposés sur une carte effectuent un choix sur le restaurant où ils souhaitent aller, parmi les 5 qui sont accessibles dans le quartier. 


Chaque restaurant a une capacité d'accueil qui reste fixe sur l'ensemble de la partie (par défaut, toutes les capacités sont à 1). 

Un épisode du jeu se déroule de la manière suivante: 

* **Phase d'initialisation**. Les joueurs sont disposés au hasard sur la ligne du bas de la carte. 

* **Phase de délibération**. Elle dure un certain nombre de pas de temps ($k$ fixé, mais à une valeur au moins aussi grande que le nombre de pas nécessaires pour un joueur pour aller au restaurant le plus loin), les joueurs peuvent se déplacer (un déplacement par pas de temps, pas de diagonale possible), observer les comportements des autres joueurs, voire se rendre dans certains restaurants. Leur comportement est guidé par leur stratégie (voir plus bas). 

* **Phase de service**. Lorsque les $k$ pas de temps sont écoulés les positions sont arrêtées et les restaurants servent de manière aléatoire le nombre de repas qui correspond à leur capacité d'accueil, parmi les joueurs qui sont dans leur restaurant. Les joueurs qui ne sont dans aucun restaurant n'obtiennent aucun plat. 

**Exemple**. Supposons par exemple que $a$, $b$ et $c$ se soient rendus dans le restaurant cooréen, qui sert 2 plats (capacité du restaurant = 2). Le premier plat est servi de manière aléatoire avec une probabilité uniforme 1/3 en $a$, $b$ et $c$. Supposons que $b$ soit sélectionné: le deuxième plat est servi avec une probabilité 1/2 entre $a$ et $c$. 

**Décompte des points**. Chaque joueur servi obtient 1 point. Puis le jeu itère sur la journée suivante. 


### Déroulement d'une partie 
Chaque joueur dispose d'une stratégie (qui peut être stochastique et adaptative) mais qui reste fixe pour l'ensemble de la partie.  
Une partie se déroule en un nombre fixe de journées. Les scores des joueurs sont les scores cumulés au cours des journées. 

### Variante "coupe-file" 

Chaque jour, un certain nombre d'objets "coupe-file" apparaissent sur la carte. Lorsqu'il dispose d'un coupe-file, le joueur est prioritaire sur les autres joueurs sans coupe-file. Si plusieurs joueurs disposent d'un coupe-file et que cela dépasse la caapcité du restaurant, le choix est fait de manière uniforme entre les joueurs qui ont un coupe-file. 
Le fait de détenir un coupe-file n'est pas observable par les autres joueurs, en particulier un joueur ne sait pas si un autre joueur possède un coupe-file parmi les autres clients d'un restaurant. 


### Hypothèses importantes 
* Les joueurs ont une **observabilité partielle de l'environnement** en ce qui concerne les autres joueurs, ce qui signifie qu'ils perçoivent seulement une région autour d'eux (que l'on pourra paramétrer, par exemple en considérant les cases autour du joueur). 
* Lorsqu'un joueur est dans un restaurant, il connait le nombre de joueurs qui sont dans le même restaurant. 
* La localisation des différents restaurants et des coupe-files s'il y en a est connaissance commune.   
* Le nombre d'itérations de la phase de délibération est connaissance commune. 
* Les joueurs ont une mémoire parfaite des évènements, c'est-à-dire qu'ils connaissent l'historique des fréquentations sur les journées précédentes.  '
* Les déplacements des joueurs ne sont pas contraints par les autres joueurs (pas de collision)




## Modules disponibles

### Module pySpriteWorld

Pour la partie graphique, vous utiliserez le module `pySpriteWorld` (développé par Yann Chevaleyre) qui s'appuie sur `pygame` et permet de manipuler simplement des personnages (sprites), cartes, et autres objets à l'écran.

Deux cartes par défaut vous sont proposées pour ce projet (`restaurant-map` et `restaurant-map2`): elles comportent 8 joueurs et 5 restaurants (en haut de la carte). Pour simplifier, on suppose que les restaurants n'occupent qu'une case. La deuxième carte comporte 2 coupe-files. 

La gestion de la carte s'opère grâce à des calques:
* un calque `background`, qui contient le fond de la carte avec les restaurants
* un calque `joueur`, où seront présents les joueurs
* un calque `ramassable`, qui contiendra les coupe-files éventuels


Les joueurs et les ramassables sont des objets Python sur lesquels vous pouvez effectuer des opérations classiques.
Par exemple, il est possible récupérer leurs coordonnées sur la carte avec `o.get_rowcol(x,y)` ou à l'inverse fixer leurs coordonnées avec `o.set_rowcol(x,y)`.
La mise à jour sur l'affichage est effective lorsque `mainiteration()` est appelé.


Notez que vous pourrez ensuite éditer vos propres cartes à l'aide de l'éditeur [Tiled](https://www.mapeditor.org/), et exporter ces cartes au format `.json`. 

Il est ensuite possible de changer la carte utilisée en modifiant le nom de la carte utilisée dans la fonction `init` du `main`:
`name = _boardname if _boardname is not None else 'restaurant-map'``

:warning: Vous n'avez pas à modifier le code de `pySpriteWorld`

### Module search

Le module `search` qui accompagne le cours est également disponible. Il permet en particulier de créer des problèmes de type grille et donc d'appeler directement certains algorithmes de recherche à base d'heuristiques vus en cours, comme A:star: pour la recherche de chemin.

## Travail demandé

### Semaine 1
**Prise en main**. A l'éxécution du fichier `main.py`, vous devez observer le comportement suivant: les joueurs sont placés au hasard sur la ligne du bas, puis ils se déplacent vers un restaurant choisi au hasard.
:point_right: votre objectif lors de cette première séance est de finaliser une partie avec cette stratégie aléatoire, c'est-à-dire d'implémenter le calcul des points et d'itérer sur les journées de la partie.

### Semaine 2 et 3
**Mise en place et test de différentes stratégies**. Il est possible de définir pour ce jeu : 
* des stratégies **non-informées**: par exemple **tétu** (aller toujours au même restaurant), **stochastique** (choisir selon une distribution de proba fixe, ce qui généralise la stratégie aléatoire de la semaine 1) 
* des stratégies **basées sur l'observation**, pendant la phase de délibération, de la situation : **greedy** (tester les restaurants, dans un ordre donné, et s'arrêter dans le premier qui ait une occupation en dessous d'un seuil donné),
* des stratégies **basées sur l'historique**, qui s'appuient sur les expériences des tours précédents: par exemple **fictitious play**, **regret-matching**

Il est évidemment possible de combiner ces différents pour obtenir des stratégies encore plus complexes. 

### Semaine 4
**Soutenances**. Celles-ci ont lieu en binôme. Vous présenterez les principaux résultats de votre projet. Il est attendu que vous compariez **au moins 6 stratégies**. Pour comparer les stratégies A vs B, vous supposerez qu'un joueur utilise la stratégie A alors que tous les autres utilisent la stratégie B. 
Le rapport doit être rédigé en markdown dans le fichier prévu à cet effet dans le répertoire `docs` (voir le template `rapport.md`).

