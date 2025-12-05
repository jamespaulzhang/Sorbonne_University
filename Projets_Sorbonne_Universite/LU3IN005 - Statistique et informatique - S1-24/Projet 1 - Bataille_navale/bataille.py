# Sam ASLO 21210657
# Yuxiang ZHANG 21202829

from copy import deepcopy
import numpy as np
from functions import *
from grille import Grille
from bateau import Bateau
from joueur import Joueur


class Bataille:
    """
        Classe de bataille navale entre deux joueurs.
    """
    def __init__(self, joueur1: Joueur, joueur2: Joueur):
        """
            Initialise une bataille avec deux joueurs.

            Parameters:
            joueur1 (Joueur): Le premier joueur.
            joueur2 (Joueur): Le deuxième joueur.
        """
        self.j1 = joueur1
        self.j2 = joueur2
        assert (self.j1.grille.taille == self.j2.grille.taille)


        self.j1_class = joueur1.__class__
        self.j2_class = joueur2.__class__
        self.original_j1_bateaux = deepcopy(joueur1.bateaux)
        self.original_j2_bateaux = deepcopy(joueur2.bateaux)

    def reset(self):
        """
        Réinitialise les grilles des deux joueurs et replace les bateaux aléatoirement.
        """
        self.j1.grille.cases = np.zeros((self.j1.grille.taille, self.j1.grille.taille), dtype=int)
        self.j2.grille.cases = np.zeros((self.j2.grille.taille, self.j2.grille.taille), dtype=int)
        self.j1.bateaux = deepcopy(self.original_j1_bateaux)
        self.j2.bateaux = deepcopy(self.original_j2_bateaux)

        self.j1 = self.j1_class(self.j1.grille, self.j1.bateaux, self.j1.name)
        self.j2 = self.j2_class(self.j2.grille, self.j2.bateaux, self.j2.name)

    def match(self) -> float:
        """
        Joue un match complet entre les deux joueurs.

        Retourne:
        float: Le nombre de tour joué (Un tour, c'est quand les deux joueurs ont joué)
        """
        joueur_actuel = self.j1
        joueur_suivant = self.j2
        tour_max = (self.j1.grille.taille * self.j1.grille.taille)
        tour_min = (np.sum([bat.size for bat in self.j1.bateaux])) # Nombre minimal de tours pour couler tous les bateaux
        tour = 0
        while tour < tour_max:
            x, y = joueur_actuel.jouer_un_coup()
            if int(tour) == tour:
                print(tour)
            tour += 0.5

            hit = joueur_suivant.grille.cases[x][y] > 0

            if hit:
                bat_num = joueur_suivant.grille.cases[x][y]
                bateau = next(bat for bat in joueur_suivant.bateaux if bat.num == bat_num) # Trouve le bateau touché
                self._print_green(f"{joueur_actuel.name} a touché un bateau!")
                joueur_actuel.enregistrer_resultat((x, y), hit)

                bateau.size -= 1
                if bateau.size == 0:
                    self._print_blue(f"le bateau {bateau.name} {joueur_suivant.name} de est coulé")
            else:
                joueur_actuel.enregistrer_resultat((x, y), hit)
                self._print_red(f"{joueur_actuel.name} a raté!")

            # Met à jour les grilles après le coup
            joueur_suivant.grille.cases[x][y] = -1
            joueur_actuel.grille_adv[x][y] = -1
            joueur_actuel.nb_coups += 1

            # Vérifier si victoire
            if tour >= tour_min and self._victoire(joueur_suivant):
                print(f"{joueur_actuel.name} a gagné!")
                return tour

            joueur_actuel, joueur_suivant = joueur_suivant, joueur_actuel
        print("match nul")
        return tour

    def _victoire(self, joueur: Joueur) -> bool:
        """
            Vérifie si tous les bateaux du joueur (l'adversaire) ont coulé.

            Parameters:
            joueur (Joueur): Le joueur à vérifier.

            Returns:
            bool: True si tous les bateaux du joueur ont été coulés, False sinon.
        """
        return not np.any(joueur.grille.cases > 0)

    def _print_red(self, text):
        print("\033[91m {}\033[00m".format(text))

    def _print_blue(self, text):
        print("\033[94m {}\033[00m".format(text))

    def _print_green(self, text):
        print("\033[92m {}\033[00m".format(text))

