# Sam ASLO 21210657
# Yuxiang ZHANG 21202829

from copy import deepcopy
import random
from typing import Optional, Tuple, List
import numpy as np
from grille import Grille
from bateau import Bateau
from functions import *

class Joueur:
    """
        Classe de base pour un joueur.
    """

    def __init__(self, grille: Grille, bateaux: list[Bateau], name: str):
        """
            Initialise un joueur.

            Parameters:
            grille (Grille): La grille de jeu.
            bateaux (list[Bateau]): La liste des bateaux.
            name (str): Le nom du joueur.
        """
        self.name = name
        self.grille = grille
        self.bateaux = bateaux
        self.nb_coups = 0

        # Génère toutes les positions possibles sur la grille et les mélange
        # utilisé par le JoueurAleatoire et le JoueurHeuristique
        self.positions_attaq = [(x, y) for x in range(grille.taille) for y in range(grille.taille)]
        random.shuffle(self.positions_attaq)

        # utilisé par le JoueurHeuristique et le JoueurProbabiliste
        self.target_positions = []
        self.last_hit = None
        self.dir = None

        # utilisé par le JoueurProbabiliste
        self.grille_adv = np.zeros((self.grille.taille, self.grille.taille), dtype=int)

        # Place les bateaux aléatoirement sur la grille
        for bateau in self.bateaux:
            placed = False
            limite = 0
            while not placed and limite < 100:
                placed = place_alea(self.grille, bateau)
                limite += 1

    def jouer_un_coup(self):
        """
            Lève une exception car cette méthode doit être implémentée par les sous-classes.
        """
        raise NotImplementedError("Joueur sans stratégie")

    def enregistrer_resultat(self, position: Tuple[np.int64, np.int64], hit: bool) -> None:
        """
            Si un coup touche la cible, ajoute en fonction du dernier coup les nouvelles positions à attaquer.
            Sinon rien n'est fait.
            Utilisé par le JoueurHeuristique et le JoueurProbabiliste.

            Paramètres :
            position (Tuple[np.int64, np.int64]) : La position du coup sous forme de tuple de coordonnées (x, y).
            hit (np.bool) : Indicateur si le coup a touché une cible.

            Retourne :
            None
        """
        if not hit:
            self.last_hit = None
            self.dir = None
            return

        x, y = position
        if self.last_hit:
            if self.dir is None:
                self.dir = self._determiner_dir(self.last_hit, position)
                if self.dir:
                    self.target_positions = self._suppr_pos(self.target_positions, self.dir)
            new_positions = self._gene_pos_dir(position, self.dir)
        else:
            new_positions = [(x + lx, y + ly) for lx, ly in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                             if 0 <= x + lx < self.grille.taille and 0 <= y + ly < self.grille.taille]
        self.target_positions.extend(new_positions)
        self.last_hit = position

    def _determiner_dir(self, last_hit: Tuple[np.int64, np.int64], curr_hit: Tuple[np.int64, np.int64]) -> Optional[int]:
        """
            Détermine la direction entre deux positions (coup actuel et dernier coup).

            Paramètres :
            last_hit (Tuple[np.int64, np.int64]) : La dernière position touchée sous forme de tuple de coordonnées (lx, ly).
            curr_hit (Tuple[np.int64, np.int64]) : La position actuelle touchée sous forme de tuple de coordonnées (cx, cy).

            Retourne :
            Optional[int]: 0 si la direction est verticale, 1 si la direction est horizontale.

        """
        lx, ly = last_hit
        cx, cy = curr_hit
        if lx == cx:
            return 0
        elif ly == cy:
            return 1

        return None

    def _gene_pos_dir(self, pos: Tuple[np.int64, np.int64], dir: int) -> List[Tuple[np.int64, np.int64]]:
        """
            Génère de nouvelles positions dans la direction donnée.

            Paramètres :
            pos (Tuple[np.int64, np.int64]) : La position actuelle sous forme de tuple de coordonnées (x, y).
            dir (int) : L'indicateur de direction (0 pour vertical, 1 pour horizontal).

            Retourne :
            List[Tuple[np.int64, np.int64]] : Une liste de nouvelles positions en fonction de la direction.
        """
        x, y = pos
        if dir is None:
            return []
        elif dir == 0:
            return [(x, y + 1), (x, y - 1)]
        elif dir == 1:
            return [(x + 1, y), (x - 1, y)]
        else:
            raise ValueError("valeur de direction invalide")

    def _suppr_pos(self, positions: Tuple[int, int], direction: int) -> List[Tuple[int, int]]:
        """
            Filtre les positions en fonction de la direction donnée.

            Parameters:
            positions (Tuple[int, int]): Liste des positions à filtrer.
            direction (int): Direction à utiliser pour le filtrage.

            Retourne :
            List[Tuple[int, int]]: Liste des positions filtrées.
        """
        if direction == 0:
            return [pos for pos in positions if pos[0] == self.last_hit[0]]
        elif direction == 1:
            return [pos for pos in positions if pos[1] == self.last_hit[1]]
        return positions


class JoueurAleatoire(Joueur):
    def __init__(self, grille: Grille, bateaux: list[Bateau], name: str):
        super().__init__(grille, bateaux, name)

    def jouer_un_coup(self):
        """
            Tirer une position parmi les positions mélangées dans la classe initialisée
        """
        return self.positions_attaq.pop()

    def enregistrer_resultat(self, position, hit):
        pass


class JoueurHeuristique(Joueur):
    def __init__(self, grille, bateaux, name):
        super().__init__(grille, bateaux, name)

    def jouer_un_coup(self):
        """
            Joue un coup en:
            sélectionnant une position parmi les positions cibles disponibles.
            et si pas des positions cibles disponibles choix aléatoire de position

            Returns:
            tuple: La position sélectionnée pour attaquer.
        """
        self.target_positions = [pos for pos in self.target_positions if pos in self.positions_attaq]

        if len(self.target_positions) >= 1:
            position = self.target_positions.pop()
            self.positions_attaq.remove(position)
            return position
        else:
            return self.positions_attaq.pop()


class JoueurProbabiliste(Joueur):
    def __init__(self, grille, bateaux, name):
        super().__init__(grille, bateaux, name)
        self.bat_adv = deepcopy(bateaux)
        self.grille_prob = np.zeros((grille.taille, grille.taille), dtype=int)

    def jouer_un_coup(self):
        """
            Joue un coup en sélectionnant une position cible ou en calculant les probabilités des positions.
            et s'il y a un coup qui a touché un bateau on suit la stratégie du JoueurHeuristique

            Retourne:
            tuple: La position sélectionnée pour attaquer.
        """
        self.target_positions = [pos for pos in self.target_positions if pos in self.positions_attaq]

        if len(self.target_positions) >= 1:
            position = self.target_positions.pop()
            self.positions_attaq.remove(position)
            return position
        else:
            return self._clc_bats_proba()

    def _clc_bat_proba(self, boat_size):
        """
            Calcule les probabilités d'attaque pour une taille de bateau donnée.
            La probabilité pour chaque cas est stockée dans la matrice grille_prob.

            Parameters:
            boat_size (int): La taille du bateau.
        """
        n = self.grille.taille
        # Parcourt horizontal
        for i in range(n):
            for j in range(n - boat_size + 1):
                if not np.any(self.grille_adv[i, j:j + boat_size] == -1):
                    self.grille_prob[i, j:j + boat_size] += 1

        # Parcourt vertical
        for i in range(n - boat_size + 1):
            for j in range(n):
                if not np.any(self.grille_adv[i:i + boat_size, j] == -1):
                    self.grille_prob[i:i + boat_size, j] += 1

    def _clc_bats_proba(self):
        """
            Calcule les probabilités d'attaque pour toutes les positions en fonction des bateaux adverses.

            Returns:
            tuple: La position avec la probabilité d'attaque la plus élevée.
        """
        self.grille_prob = np.zeros((self.grille.taille, self.grille.taille), dtype=int)

        for bat in self.bat_adv:
            self._clc_bat_proba(bat.size)

        """Dans certains cas où en raison d'une configuration des bateaux,
        certaines positions seront manquées et la probabilité des cas non
        touchés sera égale aux cas touchés égale à 0,
        cette boucle résout ce problème."""

        for i in range(self.grille.taille):
            for j in range(self.grille.taille):
                if self.grille_adv[i][j] == -1:
                    self.grille_prob[i][j] = -1

        max_index = np.argmax(self.grille_prob)

        return np.unravel_index(max_index, self.grille_prob.shape)

class JoueurMonteCarlo(Joueur):
    def __init__(self, grille: Grille, bateaux: list[Bateau], name: str, n: int = 10000000, p: float = 0.1):
        """
        Initialise un joueur utilisant l'algorithme de Monte Carlo.

        Parameters:
        grille (Grille): La grille de jeu.
        bateaux (list[Bateau]): La liste des bateaux.
        name (str): Le nom du joueur.
        n (int): Le nombre maximum d'itérations pour la simulation Monte Carlo.
        p (float): La probabilité d'échec d'un test lors de la simulation.
        """
        super().__init__(grille, bateaux, name)
        self.n = n  # Nombre maximum d'itérations
        self.p = p  # Probabilité d'échec lors de la simulation

    def jouer_un_coup(self) -> Tuple[int, int]:
        """
        Joue un coup en utilisant l'algorithme de Monte Carlo pour estimer la probabilité d'attaquer une case.

        Returns:
        Tuple[int, int]: La position sélectionnée pour attaquer.
        """
        best_position = None  # Meilleure position trouvée
        best_score = -1  # Meilleur score trouvé
        
        for position in self.positions_attaq:
            # Utiliser Monte Carlo pour évaluer la probabilité de toucher à cette position
            score = self._simuler_monte_carlo(position)  # Score basé sur la simulation
            if score > best_score:
                best_score = score
                best_position = position

        if best_position is None:
            best_position = self.positions_attaq.pop()  # Prend une position aléatoire si aucune meilleure trouvée

        return best_position  # Retourne la meilleure position

    def _simuler_monte_carlo(self, position: Tuple[int, int]) -> float:
        """
        Simule plusieurs tests Monte Carlo pour évaluer la probabilité de toucher un bateau à cette position.

        Parameters:
        position (Tuple[int, int]): La position à tester.

        Returns:
        float: Le score basé sur les résultats de la simulation Monte Carlo.
        """
        hits = 0  # Compteur de coups touchés
        
        for i in range(self.n):
            if self._est_un_hit(self.grille_adv, position):
                hits += 1  # Incrémente le compteur si c'est un hit
            elif random.random() > self.p:
                # Si l'échec se produit selon la probabilité p, on arrête ce test
                break

        return hits / self.n  # Retourne le score basé sur les hits enregistrés

    def _est_un_hit(self, grille: np.ndarray, position: Tuple[int, int]) -> bool:
        """
        Détermine si le coup à la position donnée touche un bateau.

        Parameters:
        grille (np.ndarray): La grille de l'adversaire.
        position (Tuple[int, int]): La position à tester.

        Returns:
        bool: True si le coup touche un bateau, sinon False.
        """
        x, y = position
        return grille[x, y] == -1  # Supposons que -1 indique un coup réussi

    def enregistrer_resultat(self, position: Tuple[np.int64, np.int64], hit: bool) -> None:
        """
        Enregistre le résultat d'un coup pour mettre à jour la grille de l'adversaire.

        Parameters:
        position (Tuple[np.int64, np.int64]): La position du coup.
        hit (bool): Indicateur si le coup a touché une cible.
        """
        x, y = position
        if hit:
            self.grille_adv[x, y] = -1  # Met à jour la grille en cas de hit
        else:
            self.grille_adv[x, y] = 1  # Met à jour la grille en cas de miss


class JoueurMonteCarloProbabilisteHeuristique(JoueurMonteCarlo):
    def __init__(self, grille: Grille, bateaux: list[Bateau], name: str, n: int = 10000000, p: float = 0.1):
        """
        Initialise un joueur combinant Monte Carlo, Heuristique et Probabiliste.

        Parameters:
        grille (Grille): La grille de jeu.
        bateaux (list[Bateau]): La liste des bateaux.
        name (str): Le nom du joueur.
        n (int): Le nombre maximum d'itérations pour le test Monte Carlo.
        p (float): La probabilité que la propriété soit vraie lorsqu'un test échoue.
        """
        super().__init__(grille, bateaux, name, n, p)
        self.target_positions = []  # Liste pour stocker les positions cibles pour l'heuristique
        self.bat_adv = deepcopy(bateaux)  # Copie des bateaux adverses
        self.grille_prob = np.zeros((grille.taille, grille.taille), dtype=int)  # Grille de probabilités pour le joueur probabiliste

    def jouer_un_coup(self) -> Tuple[int, int]:
        """
        Joue un coup en utilisant la stratégie combinée:
        - d'abord les positions cibles (heuristique),
        - ensuite la grille de probabilités (probabiliste),
        - enfin Monte Carlo si nécessaire.

        Returns:
        Tuple[int, int]: La position sélectionnée pour attaquer.
        """
        # Étape 1: Utiliser les positions cibles heuristiques si disponibles
        self.target_positions = [pos for pos in self.target_positions if pos in self.positions_attaq]
        if len(self.target_positions) >= 1:
            position = self.target_positions.pop()
            self.positions_attaq.remove(position)
            return position  # Retourne la première position cible heuristique trouvée

        # Étape 2: Si pas de positions cibles, utiliser la méthode probabiliste
        best_position = self._clc_bats_proba()
        if best_position:
            return best_position  # Retourne la meilleure position calculée par probabiliste

        # Étape 3: Si aucune des stratégies n'a donné un résultat, utiliser Monte Carlo
        return super().jouer_un_coup()  # Retourne une position par Monte Carlo

    def _clc_bat_proba(self, boat_size: int):
        """
        Calcule les probabilités d'attaque pour une taille de bateau donnée.
        La probabilité pour chaque case est stockée dans la matrice grille_prob.

        Parameters:
        boat_size (int): La taille du bateau.
        """
        n = self.grille.taille
        # Parcours horizontal
        for i in range(n):
            for j in range(n - boat_size + 1):
                if not np.any(self.grille_adv[i, j:j + boat_size] == -1):
                    self.grille_prob[i, j:j + boat_size] += 1  # Incrémente la probabilité pour chaque position

        # Parcours vertical
        for i in range(n - boat_size + 1):
            for j in range(n):
                if not np.any(self.grille_adv[i:i + boat_size, j] == -1):
                    self.grille_prob[i:i + boat_size, j] += 1  # Incrémente la probabilité pour chaque position

    def _clc_bats_proba(self) -> Tuple[int, int]:
        """
        Calcule les probabilités d'attaque pour toutes les positions en fonction des bateaux adverses.

        Returns:
        Tuple[int, int]: La position avec la probabilité d'attaque la plus élevée, ou None si non trouvée.
        """
        self.grille_prob = np.zeros((self.grille.taille, self.grille.taille), dtype=int)  # Réinitialise la grille de probabilité

        # Calculer les probabilités pour chaque bateau restant
        for bat in self.bat_adv:
            self._clc_bat_proba(bat.size)

        # Ignorer les positions déjà attaquées
        for i in range(self.grille.taille):
            for j in range(self.grille.taille):
                if self.grille_adv[i][j] == -1:  # Case déjà touchée
                    self.grille_prob[i][j] = -1

        max_index = np.argmax(self.grille_prob)  # Trouve l'index de la probabilité maximale
        if self.grille_prob.max() > 0:
            return np.unravel_index(max_index, self.grille_prob.shape)  # Retourne les coordonnées de la position max
        return None  # Retourne None si aucune position valable n'est trouvée

    def enregistrer_resultat(self, position: Tuple[np.int64, np.int64], hit: bool) -> None:
        """
        Enregistre le résultat d'un coup et met à jour la grille de l'adversaire.
        Si un coup touche un bateau, ajoute les cases adjacentes comme positions cibles pour de futurs coups.

        Parameters:
        position (Tuple[np.int64, np.int64]): La position du coup.
        hit (bool): Indicateur si le coup a touché une cible.
        """
        x, y = position
        if hit:
            # Met à jour la grille en cas de hit
            self.grille_adv[x, y] = -1
            # Ajouter les cases adjacentes à la liste des positions cibles
            adjacentes = [(x+dx, y+dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
            for pos in adjacentes:
                if pos in self.positions_attaq:
                    self.target_positions.append(pos)  # Ajoute les cases adjacentes comme cibles
        else:
            # Met à jour la grille en cas de miss
            self.grille_adv[x, y] = 1

        # Mettre à jour les probabilités et les cibles avec Monte Carlo
        super().enregistrer_resultat(position, hit)  # Appelle la méthode parent pour mettre à jour la grille
