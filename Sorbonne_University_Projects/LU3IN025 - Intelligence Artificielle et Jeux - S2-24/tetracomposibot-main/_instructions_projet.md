# Projet - Paint Wars

## Description

Deux équipes de 4 robots (équipes rouge et bleue) s'affrontent pour visiter au mieux une arène découpée en cases. Une case appartient à l'équipe qui l'a visitée en dernier. Le temps est limité et l'équipe qui possède le plus de cases gagne. Il s'agit là d'une variation compétitive du _problème de la patrouille multi-agents_, un problème classique en robotique.

Utilisez le fichier de configuration _config_Paintwars.py_ pour avoir un aperçu du projet.

Pour commencer, n'hésitez pas à réutiliser les comportements déjà obtenus dans les TP1 et TP2. Dans votre architecture de comportement finale, il est demandé d'avoir _a minima_:
* une architecture de subsomption ou une arbre de comportement ou un arbre de décision
* _au moins_ deux comportements type Braitenberg (ou plus avancé -- ie. vous pouvez utiliser des tests (et des boucles) à loisir)
* _au moins_ un robot utilisant exclusivement ou en partie un comportement (p.ex. Braitenberg) dont les poids ont été optimisé au préalable par algorithme génétique

Tout les coups sont permis, tant que votre code tient exclusivement dans la fonction _step_ de votre robot et que vous respectez les contraintes suivantes:
* Information interne: (1) un robot connait sont identifiant (_self.robot_id_). Utile pour éventuellement programmer des robots spécialistes. (2) Pour chaque robot, **un seul et unique entier** est autorisé comme mémoire (_self.memory_). Utile par exemple pour faire un compteur ou coder une variable d'état permettant d'alterner entre plusieurs comportements.
* Information sensorielle: les senseurs donnent des informations sur la distance à l'obstacle, mais aussi sur le type d'obstacle et, si c'est un robot, son numéro et son équipe. Utile pour créer des comportements plus complexes que de l'évitement d'obstacles.
* **Interdit**: pas de communication, pas d'information supplémentaire que celles données par les senseurs, pas de construction ou d'utilisation de cartes, pas de mémoire en plus de l'entier *memory*

## Fichiers

Fichiers importants, à modifier:
* _robot_challenger.py_: votre robot et vos stratégies.
  * def step(robotId, sensors):
    * en entrée: numéro du robot et information sensorielle
    * en sortie: renvoie vitesse de translation et vitesse de rotation
  * remarque: vous pouvez renommer ce fichier (il faudra mettre _config_Paintwars.py_ à jour).
  * __obligatoire__: donnez un nom à votre équipe (variable _team_name_), et remplissez le champs noms/prénoms/no_étudiant·e.
* _config_Paintwars.py_:
  _ display_mode_ = <type de rendu: temps réel (0), rapide (1), très rapide sans affichage (2))
  _ arena_ = <numéro de l'arène souhaitée, entre 0 et 4>
  _position_ = True ou False, permet de changer la position de départ de l'équipe Challenger (par défaut: False, à gauche)

Autres fichiers, à ne pas modifier.

* _tetracomposibot.py_: le programme principal. _Ne pas modifier._
* _arenas.py_: défini les arènes possibles. Vous pouvez éventuellement _ajouter_ des cartes pour faire des tests, si vous le souhaitez.
* _robot_champion.py_: le comportement fourni à titre d'exemple, contre lequel il va falloir faire du mieux possible! _Ne pas modifier._

Vous pouvez aussi utiliser le script _go_tournament_ qui permet de lancer 10 matches (2 matches par arène, en changeant la position de départ -- l'équipe rouge démarrera à gauche, puis à droite). Dans ce script, vous pouvez modifier la variable display_mode pour changer le mode d'affichage. Suggestion: 1 pour vitesse rapide avec une visualisation. 2 pour vitesse *très* rapide mais sans visualisation.

## Exécution

_python tetracomposibot.py config_Paintwars_ utilise par défaut les paramètres spécifiés dans _config_Paintwars.py_ (numéro de l'arène, position de départ, vitesse de rendu). Cependant, il est possible de lancer le projet avec des paramètres en ligne de commande:

_python tetracomposibot.py config_Paintwars <numero_arene> <inverser_position_de_depart> <vitesse_de_simulation>_

* <numero_arene> : entre 0 et 4
* <inverser_position_de_depart> : False ou True
* <vitesse_de_simulation> : 0 (normal), 1 (rapide), 2 (très rapide, pas d'affichage)
* Exemple: _python tetracomposibot.py config_Paintwars 1 False 1_

## Evaluation

Lors de la dernière séance de TP, vous présenterez votre travail pendant une soutenance de 15 minutes environ, en exécutant votre programme. Nous vous fournirons au début de la séance deux nouveaux fichiers:
1. _go_tournament_eval_ qui permet de lancer un tournoi sur l'ensemble des arènes initialement fournies, ainsi que de nouvelles arènes
2. _arenas_grX.py_ qui définit de nouvelles arènes inédites. le _X_ correspond à votre numéro de groupe.

Pour utiliser ces nouvelles arènes, vous devez modifier le fichier _config_Paintwars.py_ en remplaçant l'importation de _arenas.py_ par le fichier fourni _arenas_eval_grX.py_

On vous demandera:
* d'utiliser _tetracomposibot.py_ pour faire une démonstration des stratégies que vous avez implémentées;
* d'utiliser le script _go_tournament_eval_ pour présenter les scores de votre stratégie préférée contre l'équipe _robot_champion_eval_ pour chacun des labyrinthes initiaux ainsi que les labyrinthes inédits;
* d'expliquer votre architecture à l'oral, c'est à dire l'architecture globale et les comportements de base. Vous pouvez préparer une simple feuille A4 si vous souhaitez montrer visuellement votre architecture;
* de répondre aux questions qui pourront porter sur le code et sur les méthodes utilisées ou vues en cours.

Pendant la séance (hors soutenance), vous devrez vous coordonner avec les autres groupes pour faire un tournoi. Chaque groupe devra rencontrer **tous** les autres groupes adversaires, sur toutes les arènes (2 matches par arène, en variant la position de départ). Pour cela, vous modifierez _config_Paintwars.py_ pour faire s'affronter les deux équipes (i.e. une équipe jouera les bleus, l'autre les rouges -- les imports et la fonction _initialize_robots_ devront être adaptés). Vous utiliserez le script _go_tournament_eval_  pour avoir rapidement des résultats (display_mode 1 ou 2).

Vous reporterez les résultats du tournoi sur un document partagé qui vous sera donné en début de séance.

Le dernier _git push_ de votre projet avant le début de la séance d'évaluation sera pris en compte comme rendu.
