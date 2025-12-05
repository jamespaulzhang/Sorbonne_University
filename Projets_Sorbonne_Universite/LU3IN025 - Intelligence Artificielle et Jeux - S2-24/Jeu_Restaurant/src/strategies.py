import random
from search.grid2D import *

def strategie_tetue(pos_restaurants, joueur_id, choix_initiaux):
    """Stratégie têtue : le joueur va toujours au même restaurant qu'il a choisi au premier jour."""
    if joueur_id not in choix_initiaux:
        # Si c'est le premier jour, choisir un restaurant au hasard
        choix_initiaux[joueur_id] = random.choice(pos_restaurants)
    return choix_initiaux[joueur_id]

def strategie_stochastique(pos_restaurants, probabilites):
    """Stratégie stochastique : le joueur choisit un restaurant selon une distribution de probabilité."""
    return random.choices(pos_restaurants, weights=probabilites, k=1)[0]

def strategie_greedy(pos_restaurants, nb_players_in_resto, seuil, position_joueur, distance_vision, temps_restant, joueur_id, preferences, players, coupe_files):
    """
    Stratégie greedy :
    - Les joueurs ont une liste de restaurants à visiter basée sur la distance (le plus éloigné en priorité).
    - Lorsqu'un joueur entre dans un restaurant, ceux qui le voient parmi les greedy doivent le mémoriser.
    - Le joueur greedy s'approche des restaurants de sa liste à tour de rôle et mémorise le nombre de joueurs arrêtés dans chacun des restaurants.
    - Une fois tous les restaurants essayés, il choisit au hasard l'un des restaurants parmi ceux avec un nombre de joueurs en dessous de son seuil et s'y dirige, sans compter les restaurants pour lesquels il n'aura pas le temps de les rejoindre.
    """
    # Initialisation de la liste des restaurants à visiter basée sur la distance
    distances = [(r, distManhattan(position_joueur, r)) for r in pos_restaurants]
    distances.sort(key=lambda x: x[1], reverse=True)  # Tri par distance décroissante

    # Liste des restaurants à visiter
    restaurants_a_visiter = [r for r, _ in distances]

    # Mémorisation du nombre de joueurs dans chaque restaurant
    nb_joueurs_dans_resto = {tuple(r): 0 for r in pos_restaurants}

    # Observation des restaurants
    for r in restaurants_a_visiter:
        if distManhattan(position_joueur, r) <= distance_vision:
            nb_joueurs_dans_resto[tuple(r)] = nb_players_in_resto(pos_restaurants.index(r))

    # Filtrer les restaurants accessibles dans le temps restant
    restaurants_accessibles = [r for r in pos_restaurants if distManhattan(position_joueur, r) <= temps_restant]

    # Choix du restaurant
    restaurants_valides = [r for r in restaurants_accessibles if nb_joueurs_dans_resto[tuple(r)] < seuil]

    if restaurants_valides:
        choix = random.choice(restaurants_valides)
        preferences[joueur_id].append(choix)
    else:
        # Si aucun restaurant valide, choisir un restaurant au hasard parmi ceux accessibles
        if restaurants_accessibles:
            choix = random.choice(restaurants_accessibles)
            preferences[joueur_id].append(choix)
        else:
            # Si aucun restaurant accessible, rester sur place
            preferences[joueur_id].append(position_joueur)
            choix = position_joueur

    return choix

def fictitious_play(pos_restaurants, historique, joueur_id):
    """
    Stratégie ficticious : Chaque joueur suppose que ses adversaires jouent selon une distribution fixe de stratégies.
    Il observe les choix passés des autres joueurs et calcule la fréquence de chaque stratégie utilisée.
    Il joue ensuite la meilleure réponse à cette distribution de stratégies estimée.
    """
    # Initialiser un dictionnaire pour compter les visites de chaque restaurant
    restaurant_visits = {r: 0 for r in pos_restaurants}  # Initialisation des comptes de visites

    for other_id, visits in historique.items():
        if other_id != joueur_id:  # Ne prendre en compte que les autres joueurs
            for restaurant, count in visits.items():
                restaurant_visits[restaurant] += count

    # Trouver les restaurants les moins fréquentés
    min_visits = min(restaurant_visits.values())  # Nombre minimal de visites
    least_visited_restaurants = [r for r, v in restaurant_visits.items() if v == min_visits]

    return random.choice(least_visited_restaurants)  # Choisir aléatoirement un des restaurants les moins visités

def regret_matching(pos_restaurants, historique, payoffs, last_choice):
    """
    Stratégie regret_matching : Chaque joueur ajuste ses choix en fonction du regret des décisions passées.
    Le regret d'une action est la différence entre :
    - le gain qu'on aurait obtenu en jouant une autre action.
    - le gain obtenu en jouant l'action réellement choisie.
    """
    num_actions = len(pos_restaurants)
    
    # Premier tour : choix uniforme
    if sum(historique.values()) == 0:
        return random.choice(pos_restaurants)
    
    # Calculer le nombre de rounds joués
    num_rounds = sum(historique.values())
    
    # Calculer le score total obtenu jusqu'à présent
    score_total = payoffs.get(tuple(last_choice), 0) if last_choice else 0
    
    # Calculer les scores hypothétiques
    scores_hypothetiques = np.zeros(num_actions)
    for s in range(num_actions):
        scores_hypothetiques[s] = payoffs.get(tuple(pos_restaurants[s]), 0) / num_rounds
    
    # Calculer les regrets
    regrets = np.array([scores_hypothetiques[s] - score_total / num_rounds 
                       for s in range(num_actions)])
    
    # Gestion des cas initiaux où tous les regrets sont <= 0
    if np.all(regrets <= 0):
        return random.choice(pos_restaurants)
    
    # Calculer les probabilités proportionnelles aux regrets positifs
    regrets_positifs = np.maximum(regrets, 0)
    probabilites = regrets_positifs / np.sum(regrets_positifs)
    
    # Choisir l'action avec la probabilité calculée
    return pos_restaurants[np.random.choice(num_actions, p=probabilites)]

def human_behavior(pos_restaurants, nb_players_in_resto, position_joueur, distance_vision, temps_restant):
    """
    Stratégie humaine : choisit le restaurant optimal après avoir attendu pour se déplacer en fonction:
    - Du nombre de joueurs déjà présents (minimiser)
    - De la distance (minimiser)
    - Du temps restant (accessible)
    """
    available_restos = []
    
    for idx, resto in enumerate(pos_restaurants):
        distance = distManhattan(position_joueur, resto)
        # Vérifier si le restaurant est accessible dans le temps restant
        if distance <= temps_restant and distance <= distance_vision:
            nb_joueurs = nb_players_in_resto(idx)
            available_restos.append({
                'position': resto,
                'nb_joueurs': nb_joueurs,
                'distance': distance
            })
    
    if available_restos:
        # Choisir le restaurant avec le moins de joueurs, puis le plus proche
        best = min(available_restos, key=lambda x: (x['nb_joueurs'], x['distance']))
        return best['position']
    
    # Si aucun restaurant n'est accessible, choisir le plus proche (même si trop loin)
    return min(pos_restaurants, key=lambda x: distManhattan(position_joueur, x))

def strategie_imitation(pos_restaurants, historique_scores, historique_choix):
    """
    Stratégie d'imitation : le joueur examine le score total de tous les joueurs, 
    puis imite le choix de restaurant du joueur ayant le score le plus élevé.
    S'il y a plusieurs joueurs avec le score maximal, il en choisit un au hasard.
    """

    if not historique_scores:
        # S'il n'y a pas encore d'historique, choisir un restaurant au hasard
        return random.choice(pos_restaurants)

    # Trouver le score maximal parmi les joueurs
    max_score = max(historique_scores.values())
    meilleurs_joueurs = [j for j, score in historique_scores.items() if score == max_score]

    # Sélectionner au hasard l'un des joueurs ayant le score maximal
    joueur_a_mimer = random.choice(meilleurs_joueurs)

    # Retourner le dernier restaurant choisi par ce joueur, ou un restaurant au hasard s'il n'y a pas d'historique
    return historique_choix.get(joueur_a_mimer, random.choice(pos_restaurants))

def strategie_sequence_fixe(pos_restaurants, joueur_id, jour_actuel=None):
    """
    Stratégie de rotation en séquence fixe :
    - Chaque joueur suit une séquence décalée basée sur son identifiant.
    - Parcourt les restaurants dans l'ordre.
    - Une fois arrivé au dernier restaurant, recommence depuis le premier.
    """
    if jour_actuel is None:
        jour_actuel = 0  # Default to day 0 if not provided

    # Trier les restaurants pour assurer un ordre cohérent
    pos_restaurants_sorted = sorted(pos_restaurants)

    # Calculer le restaurant à visiter en fonction du jour et de l'identifiant du joueur
    index_resto = (jour_actuel + joueur_id) % len(pos_restaurants_sorted)
    resto_choisi = pos_restaurants_sorted[index_resto]

    return resto_choisi