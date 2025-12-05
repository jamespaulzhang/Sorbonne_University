import random
import numpy as np
import sys
import pygame
import matplotlib.pyplot as plt
from itertools import chain
from pySpriteWorld.gameclass import Game, check_init_game_done
from pySpriteWorld.spritebuilder import SpriteBuilder
from pySpriteWorld.players import Player
from pySpriteWorld.sprite import MovingSprite
from pySpriteWorld.ontology import Ontology
import pySpriteWorld.glo
from search.grid2D import ProblemeGrid2D
from search import probleme
from strategies import *

# ---- ---- ---- ---- ---- ----
# ---- Main                ----
# ---- ---- ---- ---- ---- ----

game = Game()

def init(_boardname=None):
    global player, game
    name = _boardname if _boardname is not None else 'restaurant-map2'
    game = Game('Cartes/' + name + '.json', SpriteBuilder)
    game.O = Ontology(True, 'SpriteSheet-32x32/tiny_spritesheet_ontology.csv')
    game.populate_sprite_names(game.O)
    game.fps = 10  # frames per second
    game.mainiteration()
    player = game.player

def item_states(items):
    return [o.get_rowcol() for o in items]

def player_states(players):
    return [p.get_rowcol() for p in players]

def main(nb_parties, nb_jours):
    iterations = 40  # nb de pas max par épisode
    if len(sys.argv) == 2:
        iterations = int(sys.argv[1])
    print("Iterations: ", iterations)

    init('restaurant-map')

    # -------------------------------
    # Initialisation
    # -------------------------------

    nb_lignes = game.spriteBuilder.rowsize
    nb_cols = game.spriteBuilder.colsize
    assert nb_lignes == nb_cols  # a priori on souhaite un plateau carre
    lMin = 2  # les limites du plateau de jeu (2 premieres lignes utilisees pour stocker les objets)
    lMax = nb_lignes - 2
    cMin = 2
    cMax = nb_cols - 2

    players = [o for o in game.layers['joueur']]
    nb_players = len(players)

    pos_restaurants = [(3, 4), (3, 7), (3, 10), (3, 13), (3, 16)]  # 5 restaurants positionnes
    nb_restos = len(pos_restaurants)
    capacity = [1] * nb_restos

    coupe_files = [o for o in game.layers["ramassable"]]  # a utiliser dans le cas de la variante coupe-file
    nb_coupe_files = len(coupe_files)

    print("lecture carte")
    print("-------------------------------------------")
    print('joueurs:', nb_players)
    print("restaurants:", nb_restos)
    print("lignes:", nb_lignes)
    print("colonnes:", nb_cols)
    print("coup_files:", nb_coupe_files)
    print("-------------------------------------------")

    def legal_position(pos):
        row, col = pos
        return ((pos not in item_states(coupe_files)) and (pos not in player_states(players)) and
                (pos not in pos_restaurants) and row > lMin and row < lMax - 1 and col >= cMin and col < cMax)

    def draw_random_location():
        while True:
            random_loc = (random.randint(lMin, lMax), random.randint(cMin, cMax))
            if legal_position(random_loc):
                return random_loc

    def players_in_resto(r):
        pos = pos_restaurants[r]
        return [i for i in range(nb_players) if players[i].get_rowcol() == pos]

    def nb_players_in_resto(r):
        return len(players_in_resto(r))

    def champ_de_vision(position_joueur, distance_vision, pos_restaurants, players):
        """
        Version modifiée avec visibilité des restaurants
        :param position_joueur: position actuelle du joueur
        :param distance_vision: distance de vision du joueur
        :param pos_restaurants: liste des positions des restaurants
        :param players: liste des objets joueurs
        :return: liste des positions visibles (restaurants, joueurs et coupe-files)
        """
        visible_positions = []

        # Ajout de la visibilité des restaurants
        for resto in pos_restaurants:
            if distManhattan(position_joueur, resto) <= distance_vision:
                visible_positions.append(resto)

        # Visibilité des joueurs
        for player in players:
            player_pos = player.get_rowcol()
            if player_pos != position_joueur and distManhattan(position_joueur, player_pos) <= distance_vision:
                visible_positions.append(player_pos)

        return visible_positions

    for o in coupe_files:
        (x1, y1) = draw_random_location()
        o.set_rowcol(x1, y1)
        game.mainiteration()

    y_init = [3, 5, 7, 9, 11, 13, 15, 17]
    x_init = 18
    random.shuffle(y_init)
    for i in range(nb_players):
        players[i].set_rowcol(x_init, y_init[i])
        game.mainiteration()

    choix_resto = [None] * nb_players
    strategies = []
    strategy_names = []
    choix_initiaux = {}
    distance_vision = float('inf')  # Définir une valeur par défaut ou laisser l'utilisateur la définir
    temps_restant = [iterations] * nb_players
    seuils = [float('inf')] * nb_players
    historique = {
        'avg_visits': {tuple(r): 1 for r in pos_restaurants}
    }
    payoffs = {i: {} for i in range(nb_players)}
    for p in range(nb_players):
        historique[p] = {tuple(r): 1 for r in pos_restaurants}  # 1 visite initiale
        payoffs[p] = {tuple(r): 1 for r in pos_restaurants}     # 1 point initial
    last_choices = [None] * nb_players
    historique_scores = {}
    historique_choix = {}
    player_coupe_file = [False] * nb_players
    preferences = [[] for _ in range(nb_players)]
    historique_choix_joueurs = [None] * nb_players

    for i in range(nb_players):
        print(f"Choisissez la stratégie pour le joueur {i+1}:")
        print("1. Stratégie têtue")
        print("2. Stratégie stochastique")
        print("3. Stratégie greedy")
        print("4. Fictitious Play")
        print("5. Regret Matching")
        print("6. Stratégie humaine")
        print("7. Stratégie d'imitation")
        print("8. Stratégie de séquence fixe")
        choice = int(input("Entrez le numéro de la stratégie : "))

        if choice == 1:
            strategies.append(lambda p=i: strategie_tetue(pos_restaurants, p, choix_initiaux))
            strategy_names.append("Têtue")
        elif choice == 2:
            probabilites = [1/nb_restos] * nb_restos
            strategies.append(lambda p=probabilites: strategie_stochastique(pos_restaurants, p))
            strategy_names.append("Stochastique")
        elif choice == 3:
            seuil = int(input(f"Entrez le seuil pour greedy (joueur {i+1}) : "))
            seuils[i] = seuil
            strategies.append(lambda p=i: strategie_greedy(
                pos_restaurants,
                nb_players_in_resto,
                seuils[p],
                players[p].get_rowcol(),
                distance_vision,
                temps_restant[p],
                p,
                preferences,
                players,  # Passé explicitement
                coupe_files  # Passé explicitement
            ))
            strategy_names.append("Greedy")
        elif choice == 4:
            strategies.append(lambda p=i: fictitious_play(pos_restaurants, historique, p))
            strategy_names.append("Fictitious Play")
        elif choice == 5:
            strategies.append(lambda p=i: regret_matching(
                [tuple(r) for r in pos_restaurants],
                historique[p],
                payoffs[p],
                last_choices[p]
            ))
            strategy_names.append("Regret Matching")
        elif choice == 6:
            strategies.append(lambda p=i: human_behavior(
                pos_restaurants,
                nb_players_in_resto,
                players[p].get_rowcol(),
                distance_vision,
                temps_restant[p],
                p,
                preferences,
                players  # Passé explicitement
            ))
            strategy_names.append("Humaine")
        elif choice == 7:
            strategies.append(lambda p=i: strategie_imitation(pos_restaurants, historique_scores, historique_choix))
            strategy_names.append("Imitation")
        elif choice == 8:
            strategies.append(lambda p=i, day_param=None: strategie_sequence_fixe(pos_restaurants, p, day_param))
            strategy_names.append("Séquence Fixe")
        else:
            print("Stratégie aléatoire par défaut.")
            strategies.append(lambda: random.choice(pos_restaurants))
            strategy_names.append("Aléatoire")

    all_scores = []

    for _ in range(nb_parties):
        total_scores = [0] * nb_players
        initial_coupe_files = [o for o in game.layers["ramassable"]]
        for day in range(nb_jours):
            print(f"\nJour {day+1}:")
            temps_restant = [iterations] * nb_players

            coupe_files = initial_coupe_files.copy()
            for o in coupe_files:
                (x1, y1) = draw_random_location()
                o.set_rowcol(x1, y1)
                game.layers["ramassable"].add(o)
                game.mainiteration()

            random.shuffle(y_init)
            for i in range(nb_players):
                players[i].set_rowcol(x_init, y_init[i])
                game.mainiteration()

            choix_resto = []
            for p, strategy in enumerate(strategies):
                if strategy_names[p] == "Séquence Fixe":
                    choix_resto.append(strategy(day_param=day))
                else:
                    choix_resto.append(strategy())
            path = []
            g = np.ones((nb_lignes, nb_cols), dtype=bool)
            for i in range(nb_lignes):
                g[0][i] = False
                g[1][i] = False
                g[nb_lignes - 1][i] = False
                g[nb_lignes - 2][i] = False
                g[i][0] = False
                g[i][1] = False
                g[i][nb_lignes - 1] = False
                g[i][nb_lignes - 2] = False

            for p in range(nb_players):
                if choix_resto[p] is None:
                    choix_resto[p] = random.choice(pos_restaurants)
                prob = ProblemeGrid2D(players[p].get_rowcol(), choix_resto[p], g, 'manhattan')
                path.append(probleme.astar(prob, verbose=False))

            player_coupe_file = [False] * nb_players

            for i in range(iterations):
                print(f"\n--- ITERATION {i+1}/{iterations} ---")

                for p in range(nb_players):
                    if i < len(path[p]):
                        (row, col) = path[p][i]
                        players[p].set_rowcol(row, col)

                        print(f"Joueur {p+1} | Position: ({row}, {col}) | Temps restant: {temps_restant[p]} | Destination: {choix_resto[p]}")

                        if temps_restant[p] <= 0:
                            print(f"ALERTE: Le temps du joueur {p+1} est écoulé !")

                            if strategy_names[p] in ["Greedy", "Humaine"]:
                                choix_resto[p] = strategies[p]()

                                if preferences[p]:
                                    new_target = preferences[p][-1]  # Utiliser le dernier choix
                                    if new_target != choix_resto[p]:
                                        print(f"Joueur {p+1} change de restaurant: {choix_resto[p]} ➝ {new_target}")
                                        choix_resto[p] = new_target
                                        prob = ProblemeGrid2D(players[p].get_rowcol(), choix_resto[p], g, 'manhattan')
                                        path[p] = probleme.astar(prob, verbose=False)

                                        if i < len(path[p]):
                                            (row, col) = path[p][i]
                                            players[p].set_rowcol(row, col)
                            elif strategy_names[p] == "Regret Matching":
                                last_choices[p] = choix_resto[p]
                                choix_resto[p] = strategies[p]()
                            else:
                                choix_resto[p] = strategies[p]()

                    for cf in coupe_files:
                        if (row, col) == cf.get_rowcol() and not player_coupe_file[p]:
                            player_coupe_file[p] = True
                            game.layers["ramassable"].remove(cf)
                            coupe_files.remove(cf)
                            print(f"Joueur {p+1} a ramassé un Coupe-file!")
                            break

                for j in range(nb_players):
                    temps_restant[j] -= 1
                    if temps_restant[j] < 0:
                        print(f"Joueur {j+1} n'a plus de temps restant !")

                game.mainiteration()
                print("-" * 40)

            scores = [0] * nb_players
            for r in range(nb_restos):
                players_here = players_in_resto(r)
                with_coupe_file = [p for p in players_here if player_coupe_file[p]]
                without_coupe_file = [p for p in players_here if not player_coupe_file[p]]
                random.shuffle(with_coupe_file)
                random.shuffle(without_coupe_file)
                served_players = with_coupe_file[:capacity[r]]
                remaining_slots = max(0, capacity[r] - len(served_players))
                served_players += without_coupe_file[:remaining_slots]
                for p in served_players:
                    scores[p] += 1

            print("Scores quotidiens :", scores)
            for p in range(nb_players):
                total_scores[p] += scores[p]

            for p in range(nb_players):
                if choix_resto[p] in pos_restaurants:
                    historique[p][tuple(choix_resto[p])] += 1

                payoffs.setdefault(p, {}).setdefault(tuple(choix_resto[p]), 0)
                payoffs[p][tuple(choix_resto[p])] += scores[p]

                historique_scores[p] = scores[p]
                historique_choix[p] = choix_resto[p]

                historique_choix_joueurs[p] = choix_resto[p]

                last_choices[p] = choix_resto[p]

        print("Scores totaux :", total_scores)
        all_scores.append(total_scores)

    strategy_total = {}
    strategy_count = {}

    for p in range(nb_players):
        strategy = strategy_names[p]
        if strategy not in strategy_total:
            strategy_total[strategy] = 0
            strategy_count[strategy] = 0
        strategy_total[strategy] += total_scores[p]
        strategy_count[strategy] += 1

    average_scores = {strategy: strategy_total[strategy] / strategy_count[strategy] if strategy_count[strategy] > 0 else 0 for strategy in strategy_total}

    labels = []
    values = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22']
    color_idx = 0

    for strategy in sorted(average_scores.keys()):
        labels.append(f"{strategy}\n(n={strategy_count[strategy]})")
        values.append(average_scores[strategy])
        color_idx += 1

    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, values, color=colors[:len(labels)])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    plt.xlabel('Stratégies (avec nombre de joueurs)')
    plt.ylabel('Score Moyen par Joueur')
    plt.title(f'Performance Comparative des Stratégies sur {nb_jours} Jours (Moyenne par Joueur)')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig('strategy_comparison.png', dpi=300)
    plt.show()

    # Plot the curve for each iteration with decision markers and player-strategy labels
    plt.figure(figsize=(12, 7))
    for i, scores in enumerate(all_scores):
        plt.plot(range(1, nb_players + 1), scores, label=f'Iteration {i+1}', marker='o')

    # Customize x-ticks to show player numbers and their strategies
    plt.xticks(range(1, nb_players + 1), [f'Joueur {j+1} : {strategy_names[j]}' for j in range(nb_players)], rotation=45)

    plt.xlabel('Joueurs et Stratégies')
    plt.ylabel('Score Total')
    plt.title(f'Scores Totaux par Joueur sur {nb_parties} parties')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig('iteration_curves.png', dpi=300)
    plt.show()

    pygame.quit()

if __name__ == '__main__':
    main(nb_parties=1, nb_jours=50)
