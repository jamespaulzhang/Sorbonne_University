# Sam ASLO 21210657
# Yuxiang ZHANG 21202829

import random
import time
from grille import Grille
from bateau import Bateau
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from typing import List

def peut_placer(grille: Grille, bateau: Bateau, position: Tuple[int, int], direction: int) -> bool:
    """
    Vérifie si un bateau peut être placé sur une grille à une position donnée et dans une direction donnée.
    direction = 1 pour horizontale, direction = 2 pour verticale.
    Renvoie True si le bateau peut être placé, sinon False.
    """
    x, y = position
    gl_taille = grille.taille
    bt_size = bateau.size

    if any([(direction == 1 and x + bt_size > gl_taille),
            (direction == 2 and y + bt_size > gl_taille)]):
        return False

    for i in range(bt_size):
        if direction == 1 and grille.cases[x + i][y] != 0:
            return False
        elif direction == 2 and grille.cases[x][y + i] != 0:
            return False

    return True


def place(grille: Grille, bateau: Bateau, position: Tuple[int, int], direction: int) -> bool:
    """
    Place un bateau sur la grille à une position donnée et dans une direction spécifiée.
    Utilise peut_placer pour vérifier si le placement est valide avant de placer le bateau.
    Renvoie True si le bateau a été placé, sinon False.
    """
    if not peut_placer(grille, bateau, position, direction):
        return False

    x, y = position
    bt_size = bateau.size

    if direction == 1:
        for i in range(bt_size):
            grille.cases[x + i, y] = bateau.num
    elif direction == 2:
        for i in range(bt_size):
            grille.cases[x, y + i] = bateau.num

    bateau.direction = direction
    bateau.position = position
    return True


def place_alea(grille: Grille, bateau: Bateau) -> bool:
    """
    Place un bateau aléatoirement sur la grille en essayant jusqu'à 100 fois s'il y a des conflits.
    Cela utilise place pour tenter de placer le bateau dans une position aléatoire.
    """
    placed = False
    nb_trys = 0
    while not placed and (nb_trys < 100):
        x = random.randint(0, grille.taille - 1)
        y = random.randint(0, grille.taille - 1)
        direction = random.randint(1, 2)
        nb_trys += 1
        placed = place(grille, bateau, (x, y), direction)
    return placed


def affiche(grille: Grille, destroy: bool) -> None:
    """
    Affiche la grille à l'aide de matplotlib avec une échelle de couleurs et des lignes de grille.
    Si destroy est True, l'affichage se met à jour après une pause, sinon il affiche l'image statiquement.
    """
    plt.imshow(grille.cases, cmap="Blues", interpolation="none")
    plt.grid(which="both", color="black", linestyle="-", linewidth=1)
    plt.xticks(np.arange(-0.5, grille.taille, 1), [])
    plt.yticks(np.arange(-0.5, grille.taille, 1), [])

    cbar = plt.colorbar()
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(["vide", "torpilleur", "sous-marin", "contre-torpilleurs", "croiseur", "porte-avions"])

    if destroy:
        plt.pause(0.5)  # Pause pour mettre à jour l'affichage
        plt.clf()  # Efface la figure pour la prochaine itération
    else:
        plt.show()  # Affiche la figure


def grilles_eq(grilleA: Grille, grilleB: Grille) -> bool:
    """
    Compare deux grilles pour vérifier si elles sont identiques.
    Renvoie True si les deux grilles sont égales, sinon False.
    """
    return np.array_equal(grilleA.cases, grilleB.cases)


def genere_grille() -> Grille:
    """
    Génère une grille avec des bateaux placés aléatoirement.
    Retourne une instance de la grille avec tous les bateaux déjà placés.
    """
    grille = Grille(10)
    bateaux_noms = ["porte-avions", "croiseur", "contre-torpilleurs", "sous-marin", "torpilleur"]
    for nom in bateaux_noms:
        bateau = Bateau(nom)
        place_alea(grille, bateau)
    return grille


def nb_placer(grille: Grille, bateau: Bateau) -> int:
    """
    Calcule le nombre de façons possibles de placer un bateau donné sur une grille.
    Retourne ce nombre basé sur la taille de la grille et la taille du bateau.
    """
    gt_taille = grille.taille
    bt_size = bateau.size
    return 2 * (gt_taille - bt_size + 1) * gt_taille


def nb_total_placer(grille: Grille, bateaux: List[Bateau]) -> int:
    """
    Calcule le nombre total de façons possibles de placer une liste de bateaux sur la grille.
    Retourne ce nombre en multipliant les possibilités de chaque bateau.
    """
    total = 1
    for bateau in bateaux:
        total *= nb_placer(grille, bateau)
    return total

def nb_grilles(grille: Grille) -> int:
    """
    Calcule combien de tentatives sont nécessaires pour générer une grille identique à une grille de référence.
    Affiche le nombre d'essais jusqu'à ce que les deux grilles soient égales.
    """
    nb_try = 0
    while not (grilles_eq(grille, genere_grille())):
        nb_try += 1
    return nb_try


# algorithme
"""
1. Condition de terminaison: Si tous les bateaux ont été placés (`index == len(bateaux)`), l'algorithme retourne 1, indiquant une configuration valide.
2. Le bateau actuel à placer est récupéré de la liste des bateaux à l'index donné.
3. On parcourt toutes les positions possibles `(x, y)` sur la grille.
4.Pour chaque position, on tente de placer le bateau. Si le placement réussit :
    - On appele la fonction récursivement pour placer le bateau suivant.
    - Le bateau est ensuite retiré de la grille.

Cet algorithme suppose que tous les bateaux peuvent être placés seulemnt dans la même direction, ce qui simplifie les calculs.
Nous calculons donc toutes les différentes configurations dans une direction puis on renvoi le nombre multiplié par deux
Le problème avec cet algorithme est qu'il ne prend pas en compte le placement des bateaux dans les deux directions.
Et si la taille du grill ou la liste des bateaux est grande les calculs deviennent impossibles.

"""

def remove(grille: Grille, bateau: Bateau) -> None:
    """
    Retire un bateau de la grille à partir de sa position et direction actuelle.
    Met à jour les cases correspondantes de la grille en les remplaçant par des zéros.
    """
    x, y = bateau.position
    bt_size = bateau.size
    direction = bateau.direction

    if direction == 1:
        for i in range(bt_size):
            grille.cases[x + i, y] = 0
    elif direction == 2:
        for i in range(bt_size):
            grille.cases[x, y + i] = 0


def count_configs(grille: Grille, bateaux: List[Bateau], index: int = 0) -> int:
    """
    Compte le nombre total de configurations valides pour placer une liste de bateaux sur la grille.
    Utilise un algorithme récursif pour parcourir toutes les positions possibles pour chaque bateau.
    Retourne le nombre de configurations possibles.
    """
    if index == len(bateaux):
        return 1

    bateau = bateaux[index]
    ways = 0

    for x in range(grille.taille):
        for y in range(grille.taille):
            if place(grille, bateau, (x, y), 1):
                # pour afficher l'animation
                # affiche(grille, True)
                # time.sleep(0.5)
                ways += count_configs(grille, bateaux, index + 1)
                remove(grille, bateau)
    return 2 * ways

if __name__ == "__main__":
    main()
