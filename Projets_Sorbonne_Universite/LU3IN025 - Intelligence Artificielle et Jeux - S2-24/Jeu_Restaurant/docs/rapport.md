# Rapport de Projet : Jeu des Restaurants

# Groupe
Yuxiang ZHANG
Antoine LECOMTE

## Introduction
Dans le cadre de ce projet, nous avons développé un jeu stratégique où des joueurs concurrents doivent choisir quotidiennement un restaurant pour maximiser leurs chances d'être servis. Le système intègre plusieurs stratégies d'IA et permet d'analyser leurs performances sur 50 journées simulées.

## Description des choix importants d'implémentation

### Architecture technique
- **Bibliothèques utilisées** :
  - `pySpriteWorld` pour la visualisation
  - `search` pour les algorithmes de pathfinding
  - `matplotlib` pour l'analyse des résultats
- **Structure du code** :
  - Module `strategies.py` contenant les implémentations des stratégies
  - Script principal `main.py` gérant la simulation
  - Cartes personnalisées au format JSON

### Mécaniques clés
1. **Déplacement** : A* pour le chemin optimal
2. **Vision** : Champ de vision Manhattan (rayon=5) pour les joueurs utilisant greedy & greedy_complex
3. **Coupe-files** : Priorité aléatoire avec objets ramassables
4. **Score** : Cumul quotidien avec capacités de restaurants

## Description des stratégies proposées

### 1. Stratégie Têtue
**Mécanique** :
- Attribution aléatoire d'un restaurant au premier jour
- Même choix quotidien indépendamment des conditions
- Évite la compétition dynamique

### 2. Stratégie Stochastique
**Mécanique** :
- Distribution contrôlée via `probabilites`
- Permet d'éviter les schémas prévisibles
- Adaptable via ajustement des poids (ex: favoriser les restaurants peu fréquentés)

### 3. Stratégie Greedy
**Mécanique** :
- Trie les restaurants par distance Manhattan décroissante
- **Recalcul dynamique** :
  - Vérifie en temps réel la capacité (`seuil`) et l'accessibilité (`temps_restant`)
  - Met à jour les préférences via `preferences[joueur_id]`
- **Système de repli** :
  1. Restaurants sous le seuil dans le champ de vision
  2. Restaurants accessibles en temps restant
  3. Position actuelle en dernier recours

### 4. Fictitious Play
**Mécanique** :
- Calcule les fréquences historiques des choix adverses
- Cible les restaurants les moins visités (min_visits)
- Évite les stratégies majoritaires via l'historique global

### 5. Regret Matching
**Mécanique** :
- Compare les gains réels vs hypothétiques
- **Mise à jour adaptative** :
  - Probabilités proportionnelles aux regrets positifs
  - Randomisation uniforme si tous regrets ≤ 0

### 6. Stratégie Humaine
**Améliorations par rapport à Greedy** :
- Observation de la situation et choix optimal
- Combine distance, visibilité et comportements adverses
(Stratégie difficile à mettre en place)

### 7. Stratégie d'Imitation
**Mécanique** :
- Identifie les meilleurs joueurs avec `max_score`
- Copie le dernier choix des meilleurs sans analyse contextuelle
- Risque de saturation des restaurants "populaires" : les plus souvent visités

### 8. Stratégie Séquence Fixe
**Mécanique** :
- Parcours cyclique des restaurants triés
- Séquence unique par joueur via `joueur_id`
- Garantit une répartition temporelle uniforme

## Description des résultats

Précisions :
- Nous avons utilisé la carte 1 (sans coupe-files) pour l'ensemble des graphes, de sorte à ne pas tronquer les résultats.
- Nous avons utilisé un seuil de 1 pour l'ensemble des tests des stratégies greedy et human behavior, ainsi qu'un champ de vision de 5 cases pour greedy et infini pour human behavior, afin de simuler des conditions optimales d'observation.

### Analyse comparative

| **Stratégie (1 joueur)** | **Adversaire (7 joueurs)** | **Score (1 joueur)** | **Score (7 joueurs)** | **Résultat** | **Explication** | **Image** |
|---------------------------|----------------------------|-----------------------|------------------------|--------------|-----------------|-----------------|
| **Têtue**                 | Stochastique               | 26.00                 | 26.57                  | Défaite       | La stratégie têtue est légèrement moins performante car elle ne s'adapte pas aux variations de fréquentation des restaurants, contrairement à la stratégie stochastique qui peut ajuster ses choix en fonction des probabilités. | ![Comparaison 1 Têtue vs 7 Stochastique](./graphes/1Tetue_VS_7Stochastique.png) |
| **Têtue**                 | Greedy                    | 28.00                 | 30.43                  | Défaite       | La stratégie greedy exploite mieux les opportunités en temps réel, ce qui lui permet de maximiser ses chances d'être servie, tandis que la stratégie têtue reste fixe et ne profite pas des variations. | ![Comparaison 1 Têtue vs 7 Greedy](./graphes/1Tetue_VS_7Greedy.png) |
| **Têtue**                 | Fictitious Play            | 50.00                 | 27.14                  | Victoire      | La stratégie têtue excelle ici car elle évite la compétition directe en restant fidèle à un restaurant, tandis que Fictitious Play s'adapte aux choix passés des autres joueurs, ce qui peut conduire à une saturation des restaurants. | ![Comparaison 1 Têtue vs 7 Fictitious Play](./graphes/1Tetue_VS_7FP.png) |
| **Têtue**                 | Regret Matching            | 30.00                 | 28.00                  | Victoire      | La stratégie têtue bénéficie de sa constance, tandis que Regret Matching peut être désavantagée par des ajustements fréquents qui ne sont pas toujours optimaux. | ![Comparaison 1 Têtue vs 7 Regret Matching](./graphes/1Tetue_VS_7RM.png) |
| **Têtue**                 | Imitation                  | 14.57                 | 13.00                  | Victoire      | Les deux stratégies obtiennent des scores faibles en raison d'une saturation des restaurants populaires. Imitation souffre particulièrement car elle copie les choix des joueurs ayant les scores les plus élevés, ce qui peut mener à une surpopulation des mêmes restaurants. | ![Comparaison 1 Têtue vs 7 Imitation](./graphes/1Tetue_VS_7Imitation.png) |
| **Têtue**                 | Séquence Fixe              | 21.00                 | 32.71                  | Défaite       | La séquence fixe répartit mieux les joueurs sur les restaurants, réduisant ainsi la concurrence directe, tandis que la stratégie têtue reste fixe et peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Têtue vs 7 Séquence Fixe](./graphes/1Tetue_VS_7SF.png) |
| **Stochastique**          | Têtue                      | 22.00                 | 27.00                  | Défaite       | La stratégie têtue bénéficie de sa constance, tandis que la stratégie stochastique peut se retrouver dans des restaurants surpeuplés en raison de sa nature aléatoire. | ![Comparaison 1 Stochastique vs 7 Têtue](./graphes/1Stochastique_VS_7Tetue.png) |
| **Stochastique**          | Greedy                     | 19.00                 | 32.00                  | Défaite       | Greedy exploite mieux les opportunités en temps réel, tandis que la stratégie stochastique manque de cohérence dans ses choix. | ![Comparaison 1 Stochastique vs 7 Greedy](./graphes/1Stochastique_VS_7Greedy.png) |
| **Stochastique**          | Fictitious Play            | 24.00                 | 30.00                  | Défaite       | Fictitious Play s'adapte mieux aux choix passés des autres joueurs, ce qui lui permet de maximiser ses chances d'être servie. | ![Comparaison 1 Stochastique vs 7 Fictitious Play](./graphes/1Stochastique_VS_7FP.png) |
| **Stochastique**          | Regret Matching            | 22.00                 | 27.00                  | Défaite       | La stratégie Regret Matching tire parti de son processus d’ajustement basé sur les regrets, lui permettant de converger vers des choix plus optimaux au fil du temps. En revanche, la stratégie stochastique, bien que plus imprévisible, peut avoir du mal à capitaliser sur les opportunités les plus rentables de manière cohérente. | ![Comparaison 1 Stochastique vs 7 Regret Matching](./graphes/1Stochastique_VS_7RM.png) |
| **Stochastique**          | Imitation                  | 31.00                 | 22.57                  | Victoire      | Imitation souffre de la saturation des restaurants populaires, tandis que la stratégie stochastique peut mieux répartir ses choix. | ![Comparaison 1 Stochastique vs 7 Imitation](./graphes/1Stochastique_VS_7Imitation.png) |
| **Stochastique**          | Séquence Fixe              | 29.00                 | 31.57                  | Défaite       | La séquence fixe répartit mieux les joueurs, réduisant ainsi la concurrence directe, tandis que la stratégie stochastique peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Stochastique vs 7 Séquence Fixe](./graphes/1Stochastique_VS_7SF.png) |
| **Greedy**                | Têtue                      | 31.00                 | 27.14                  | Victoire      | Greedy exploite mieux les opportunités en temps réel, tandis que la stratégie têtue reste fixe et ne profite pas des variations. | ![Comparaison 1 Greedy vs 7 Têtue](./graphes/1Greedy_VS_7Tetue.png) |
| **Greedy**                | Stochastique               | 28.00                 | 25.57                  | Victoire      | Greedy bénéficie de sa capacité à recalculer dynamiquement ses choix, tandis que la stratégie stochastique manque de cohérence. | ![Comparaison 1 Greedy vs 7 Stochastique](./graphes/1Greedy_VS_7Stochastique.png) |
| **Greedy**                | Fictitious Play            | 23.00                 | 31.86                  | Défaite       | Fictitious Play s'adapte mieux aux choix passés des autres joueurs, ce qui lui permet de maximiser ses chances d'être servie. | ![Comparaison 1 Greedy vs 7 Fictitious Play](./graphes/1Greedy_VS_7FP.png) |
| **Greedy**                | Regret Matching            | 30.00                 | 25.57                  | Victoire      | La stratégie Greedy, en privilégiant systématiquement les choix les plus avantageux à court terme, parvient à surpasser Regret Matching. Bien que cette dernière stratégie s’adapte progressivement en fonction des regrets passés, son ajustement peut être trop lent face à une approche plus opportuniste et immédiate comme Greedy. | ![Comparaison 1 Greedy vs 7 Regret Matching](./graphes/1Greedy_VS_7RM.png) |
| **Greedy**                | Imitation                  | 29.00                 | 21.00                  | Victoire      | Imitation souffre de la saturation des restaurants populaires, tandis que Greedy peut mieux répartir ses choix. | ![Comparaison 1 Greedy vs 7 Imitation](./graphes/1Greedy_VS_7Imitation.png) |
| **Greedy**                | Séquence Fixe              | 22.00                 | 32.57                  | Défaite       | La séquence fixe répartit mieux les joueurs, réduisant ainsi la concurrence directe, tandis que Greedy peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Greedy vs 7 Séquence Fixe](./graphes/1Greedy_VS_7SF.png) |
| **Fictitious Play**       | Têtue                      | 50.00                 | 21.90                  | Victoire      | Fictitious Play s'adapte mieux aux choix passés des autres joueurs, tandis que la stratégie têtue reste fixe et peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Fictitious Play vs 7 Têtue](./graphes/1FP_VS_7Tetue.png) |
| **Fictitious Play**       | Stochastique               | 18.00                 | 27.86                  | Défaite       | La stratégie stochastique bénéficie de sa variabilité, tandis que Fictitious Play peut être désavantagée par des ajustements fréquents qui ne sont pas toujours optimaux. | ![Comparaison 1 Fictitious Play vs 7 Stochastique](./graphes/1FP_VS_7Stochastique.png) |
| **Fictitious Play**       | Greedy                     | 27.00                 | 31.29                  | Défaite       | Greedy exploite mieux les opportunités en temps réel, tandis que Fictitious Play peut être moins réactif. | ![Comparaison 1 Fictitious Play vs 7 Greedy](./graphes/1FP_VS_7Greedy.png) |
| **Fictitious Play**       | Regret Matching            | 35.00                 | 26.57                  | Victoire      | Fictitious Play s'adapte mieux aux choix passés des autres joueurs, tandis que Regret Matching peut être désavantagée par des ajustements fréquents. | ![Comparaison 1 Fictitious Play vs 7 Regret Matching](./graphes/1FP_VS_7RM.png) |
| **Fictitious Play**       | Imitation                  | 32.00                 | 20.86                  | Victoire      | Imitation souffre de la saturation des restaurants populaires, tandis que Fictitious Play peut mieux répartir ses choix. | ![Comparaison 1 Fictitious Play vs 7 Imitation](./graphes/1FP_VS_7Imitation.png) |
| **Fictitious Play**       | Séquence Fixe              | 16.00                 | 33.43                  | Défaite       | La séquence fixe répartit mieux les joueurs, réduisant ainsi la concurrence directe, tandis que Fictitious Play peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Fictitious Play vs 7 Séquence Fixe](./graphes/1FP_VS_7SF.png) |
| **Regret Matching**       | Têtue                      | 37.00                 | 26.86                  | Victoire      | Regret Matching ajuste mieux ses choix en fonction des regrets passés, tandis que la stratégie têtue reste fixe et peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Regret Matching vs 7 Têtue](./graphes/1RM_VS_7Tetue.png) |
| **Regret Matching**       | Stochastique               | 27.00                 | 25.86                  | Victoire      | La stratégie stochastique bénéficie de sa variabilité, mais Regret Matching parvient à s’adapter progressivement en exploitant les choix les plus performants sur le long terme. | ![Comparaison 1 Regret Matching vs 7 Stochastique](./graphes/1RM_VS_7Stochastique.png) |
| **Regret Matching**       | Greedy                     | 29.00                 | 25.29                  | Victoire      | Regret Matching ajuste mieux ses choix en fonction des regrets passés, tandis que Greedy peut être moins réactif. | ![Comparaison 1 Regret Matching vs 7 Greedy](./graphes/1RM_VS_7Greedy.png) |
| **Regret Matching**       | Fictitious Play            | 18.00                 | 30.71                  | Défaite       | Fictitious Play s'adapte mieux aux choix passés des autres joueurs, tandis que Regret Matching peut être désavantagée par des ajustements fréquents. | ![Comparaison 1 Regret Matching vs 7 Fictitious Play](./graphes/1RM_VS_7FP.png) |
| **Regret Matching**       | Imitation                  | 28.00                 | 21.43                  | Victoire      | Imitation souffre de la saturation des restaurants populaires, tandis que Regret Matching peut mieux répartir ses choix. | ![Comparaison 1 Regret Matching vs 7 Imitation](./graphes/1RM_VS_7Imitation.png) |
| **Regret Matching**       | Séquence Fixe              | 28.00                 | 31.71                  | Défaite       | La séquence fixe répartit mieux les joueurs, réduisant ainsi la concurrence directe, tandis que Regret Matching peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Regret Matching vs 7 Séquence Fixe](./graphes/1RM_VS_7SF.png) |
| **Imitation**             | Têtue                      | 21.00                 | 32.71                  | Défaite       | La stratégie têtue bénéficie de sa constance, tandis qu'Imitation souffre de la saturation des restaurants populaires. | ![Comparaison 1 Imitation vs 7 Têtue](./graphes/1Imitation_VS_7Tetue.png) |
| **Imitation**             | Stochastique               | 24.00                 | 26.00                  | Défaite       | La stratégie stochastique bénéficie de sa variabilité, tandis qu'Imitation peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Imitation vs 7 Stochastique](./graphes/1Imitation_VS_7Stochastique.png) |
| **Imitation**             | Greedy                     | 25.00                 | 30.43                  | Défaite       | Greedy exploite mieux les opportunités en temps réel, tandis qu'Imitation peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Imitation vs 7 Greedy](./graphes/1Imitation_VS_7Greedy.png) |
| **Imitation**             | Fictitious Play            | 22.00                 | 31.43                  | Défaite       | Fictitious Play s'adapte mieux aux choix passés des autres joueurs, tandis qu'Imitation peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Imitation vs 7 Fictitious Play](./graphes/1Imitation_VS_7FP.png) |
| **Imitation**             | Regret Matching            | 31.00                 | 26.43                  | Victoire      | La stratégie Imitation profite de l'observation des meilleurs joueurs pour reproduire leurs choix, lui permettant ainsi d’optimiser ses décisions plus rapidement que Regret Matching. Ce dernier, bien qu’adaptatif, ajuste ses choix de manière plus progressive, ce qui peut le désavantager face à une approche qui exploite directement les succès des autres. | ![Comparaison 1 Imitation vs 7 Regret Matching](./graphes/1Imitation_VS_7RM.png) |
| **Imitation**             | Séquence Fixe              | 18.00                 | 33.14                  | Défaite       | La séquence fixe répartit mieux les joueurs, réduisant ainsi la concurrence directe, tandis qu'Imitation peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Imitation vs 7 Séquence Fixe](./graphes/1Imitation_VS_7FP.png) |
| **Séquence Fixe**         | Têtue                      | 28.00                 | 20.29                  | Victoire      | La séquence fixe répartit mieux les joueurs, réduisant ainsi la concurrence directe, tandis que la stratégie têtue peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Séquence Fixe vs 7 Têtue](./graphes/1SF_VS_7Tetue.png) |
| **Séquence Fixe**         | Stochastique               | 26.00                 | 25.43                  | Victoire      | La séquence fixe répartit mieux les joueurs, tandis que la stratégie stochastique peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Séquence Fixe vs 7 Stochastique](./graphes/1SF_VS_7Stochastique.png) |
| **Séquence Fixe**         | Greedy                     | 32.00                 | 25.71                  | Victoire      | La séquence fixe répartit mieux les joueurs, tandis que Greedy peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Séquence Fixe vs 7 Greedy](./graphes/1SF_VS_7Greedy.png) |
| **Séquence Fixe**         | Fictitious Play            | 21.00                 | 31.43                  | Défaite       | Fictitious Play s'adapte mieux aux choix passés des autres joueurs, tandis que la séquence fixe peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Séquence Fixe vs 7 Fictitious Play](./graphes/1SF_VS_7FP.png) |
| **Séquence Fixe**         | Regret Matching            | 24.00                 | 26.14                  | Défaite       | Regret Matching ajuste mieux ses choix en fonction des regrets passés, tandis que la séquence fixe peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Séquence Fixe vs 7 Regret Matching](./graphes/1SF_VS_7RM.png) |
| **Séquence Fixe**         | Imitation                  | 32.00                 | 21.29                  | Victoire      | La séquence fixe répartit mieux les joueurs, tandis qu'Imitation peut se retrouver dans des restaurants surpeuplés. | ![Comparaison 1 Séquence Fixe vs 7 Imitation](./graphes/1SF_VS_7Imitation.png) |

### Comparaison entre toutes les Stratégies
![Comparaison entre toutes les Stratégies](./graphes/scores_finaux_3_parties.png)
![Histogramme de la partie 3](./graphes/all_strategies_partie_3.png)
- **Têtue** : 25.00 points
- **Stochastique** : 23.00 points
- **Greedy** : 31.00 points
- **Humaine** : 44.00 points
- **Fictitious Play** : 29.00 points
- **Regret Matching** : 31.00 points
- **Imitation** : 26.00 points
- **Séquence Fixe** : 31.00 points
- **Explication** : La stratégie humaine obtient le score le plus élevé en raison de l'observation en temps réel des déplacements des autres joueurs, tandis que les stratégies têtues, stochastique et regret matching obtiennent des scores très moyens, il y a trop peu d'itérations mais avec davantage regret matching est très performante grâce à sa capacité à s'adapter aux variations de fréquentation. La séquence fixe et greedy obtiennent des scores très proches malgré la différence (programmé à l'avance pour SF contre stratégie d'observation pour greedy) avec des scores plus élevés. Enfin ficticious les dépasse et semble avec un score plus ou moins stable entre les parties, bien qu'elle reste très largement en dessous des performances de la stratégie reproduisant une réflexion humaine.


### Conclusion
Dans ce projet, nous avons exploré plusieurs stratégies pour maximiser les chances d'un joueur d'être servi dans un jeu stratégique de restaurants. Les résultats montrent que chaque stratégie a ses avantages et inconvénients en fonction des conditions du jeu.

- **Stratégie Têtue** : Cette stratégie obtient le score le plus élevé en raison de sa constance. Elle évite la compétition directe en restant fidèle à un restaurant, ce qui est particulièrement efficace lorsque les autres joueurs utilisent des stratégies plus dynamiques.

- **Stratégie Stochastique** : Bien que flexible, cette stratégie peut être désavantagée par des ajustements fréquents qui ne sont pas toujours optimaux. Elle est plus performante dans des environnements imprévisibles.

- **Stratégie Greedy** : Cette stratégie exploite mieux les opportunités en temps réel, ce qui lui permet de maximiser ses chances d'être servie. Cependant, elle peut être moins réactive face à des changements soudains dans la fréquentation des restaurants.

- **Stratégie Humaine** : Bien que plus complexe à mettre en œuvre, cette stratégie permet une meilleure répartition des joueurs en tenant compte de la visibilité et des comportements adverses. Elle est particulièrement efficace dans des environnements dynamiques et avec suffisamment d'itérations pour avoir le temps de voir les joueurs s'arrêter.

- **Fictitious Play** : Cette stratégie s'adapte mieux aux choix passés des autres joueurs, ce qui lui permet de maximiser ses chances d'être servie. Cependant, elle peut être désavantagée par des ajustements fréquents qui ne sont pas toujours optimaux.

- **Regret Matching** : Cette stratégie ajuste mieux ses choix en fonction des regrets passés, ce qui lui permet de maximiser ses chances. Elle est particulièrement efficace dans des environnements compétitifs.

- **Stratégie d'Imitation** : Bien que simple à implémenter, cette stratégie peut souffrir de la saturation des restaurants populaires. Elle est plus performante lorsque les meilleurs joueurs ont des scores élevés.

- **Séquence Fixe** : Cette stratégie répartit mieux les joueurs sur les restaurants, réduisant ainsi la concurrence directe. Elle est particulièrement efficace lorsque les autres joueurs utilisent des stratégies plus dynamiques.

En conclusion, il n'existe pas de stratégie universellement optimale. Le choix de la stratégie dépend des conditions spécifiques du jeu, telles que la fréquentation des restaurants, le comportement des autres joueurs, et la capacité d'adaptation de la stratégie. En affrontant toutes les autres, la stratégie Humaine est optimale, mais elle reste soumise à certaines contraintes comme la durée des manches et le champ de vision limité. Les stratégies adaptatives Greedy, Fictitious Play et Regret Matching obtiennent des scores compétitifs grâce à leur capacité à s'adapter aux variations de fréquentation, mais Regret Matching s'avère moins efficace lorsqu'il y a peu d'itérations, tandis que les stratégies plus statiques comme Têtue, Séquence Fixe et les stratégies difficiles à prévoir Stochastique et Imitation obtiennent des scores qui peuvent varier davantage en raison de la saturation des restaurants populaires et/ou du facteur aléatoire de la stratégie.