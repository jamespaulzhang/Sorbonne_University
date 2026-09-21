# Sam ASLO 21210657
# Yuxiang ZHANG 21202829

import matplotlib.pyplot as plt
import numpy as np
from grille import Grille
from bateau import Bateau
from joueur import JoueurAleatoire
from joueur import JoueurHeuristique
from joueur import JoueurProbabiliste
from joueur import JoueurMonteCarlo
from joueur import JoueurMonteCarloProbabilisteHeuristique
from bataille import Bataille
from typing import List

def comparer_strategies(n_simulations: int, bataille: Bataille) -> List[float]:
    """
        Simule plusieurs parties de bataille et enregistre le nombre de coups nécessaires pour chaque partie.

        Args:
            n_simulations (int): Le nombre de simulations à exécuter.
            bataille (Bataille): Une instance de la classe Bataille, représentant le jeu de bataille.

        Retourne:
            List[float]: Une liste contenant le nombre de coups nécessaires pour chaque simulation.
    """
    coups = []
    for _ in range(n_simulations):
        coups.append(bataille.match())
        bataille.reset()
    return coups


def test_strategie(joueur1, joueur2, nb_bataille: int) -> List[float]:
    """
    Teste une stratégie donnée et retourne le nombre de coups.

    Args:
        joueur1: Le joueur 1.
        joueur2: Le joueur 2.
        nb_bataille (int): Le nombre de batailles à simuler.

    Returns:
        List[float]: Une liste contenant le nombre de coups pour chaque bataille.
    """
    bataille = Bataille(joueur1, joueur2)
    return comparer_strategies(nb_bataille, bataille)


def menu():
    print("Choisissez une option pour tester les stratégies:")
    print("1. Tester Joueur Aleatoire")
    print("2. Tester Joueur Heuristique")
    print("3. Tester Joueur Probabiliste")
    print("4. Tester Joueur Monte Carlo")
    print("5. Tester Toutes les Stratégies")
    print("0. Quitter")


def main():
    nb_bataille = 100

    while True:
        menu()
        choix = input("Entrez votre choix (0-5): ")

        if choix == '0':
            print("Au revoir!")
            break
        elif choix == '1':
            # Test JoueurAleatoire
            grille1 = Grille(10)
            grille2 = Grille(10)
            bateaux = [Bateau("torpilleur"), Bateau("sous-marin"), Bateau("contre-torpilleurs"), Bateau("croiseur"), Bateau("porte-avions")]
            joueur1 = JoueurAleatoire(grille1, bateaux, "Player 1")
            joueur2 = JoueurAleatoire(grille2, bateaux, "Player 2")
            aleatoire_coups = test_strategie(joueur1, joueur2, nb_bataille)

            plt.hist(aleatoire_coups, bins=20, color='blue', alpha=0.5, label='Stratégie aléatoire')
            plt.title("Stratégie Aléatoire")
            plt.xlabel("Nombre de coups")
            plt.ylabel("Fréquence")
            plt.legend()
            plt.show()
            print(f"Espérance aléatoire: {np.mean(aleatoire_coups)}")

        elif choix == '2':
            # Test JoueurHeuristique
            grille1 = Grille(10)
            grille2 = Grille(10)
            bateaux = [Bateau("torpilleur"), Bateau("sous-marin"), Bateau("contre-torpilleurs"), Bateau("croiseur"), Bateau("porte-avions")]
            joueur1 = JoueurHeuristique(grille1, bateaux, "Player 1")
            joueur2 = JoueurHeuristique(grille2, bateaux, "Player 2")
            heuristique_coups = test_strategie(joueur1, joueur2, nb_bataille)

            plt.hist(heuristique_coups, bins=20, color='green', alpha=0.5, label='Stratégie heuristique')
            plt.title("Stratégie Heuristique")
            plt.xlabel("Nombre de coups")
            plt.ylabel("Fréquence")
            plt.legend()
            plt.show()
            print(f"Espérance heuristique: {np.mean(heuristique_coups)}")

        elif choix == '3':
            # Test JoueurProbabiliste
            grille1 = Grille(10)
            grille2 = Grille(10)
            bateaux = [Bateau("torpilleur"), Bateau("sous-marin"), Bateau("contre-torpilleurs"), Bateau("croiseur"), Bateau("porte-avions")]
            joueur1 = JoueurProbabiliste(grille1, bateaux, "Player 1")
            joueur2 = JoueurProbabiliste(grille2, bateaux, "Player 2")
            probabiliste_coups = test_strategie(joueur1, joueur2, nb_bataille)

            plt.hist(probabiliste_coups, bins=20, color='red', alpha=0.5, label='Stratégie probabiliste')
            plt.title("Stratégie Probabiliste")
            plt.xlabel("Nombre de coups")
            plt.ylabel("Fréquence")
            plt.legend()
            plt.show()
            print(f"Espérance probabiliste: {np.mean(probabiliste_coups)}")

        elif choix == '4':
            # Test JoueurMonteCarlo
            grille1 = Grille(10)
            grille2 = Grille(10)
            bateaux = [Bateau("torpilleur"), Bateau("sous-marin"), Bateau("contre-torpilleurs"), Bateau("croiseur"), Bateau("porte-avions")]
            joueur1 = JoueurMonteCarloProbabilisteHeuristique(grille1, bateaux, "Monte Carlo Player")
            joueur2 = JoueurMonteCarloProbabilisteHeuristique(grille2, bateaux, "Monte Carlo Player 2")
            montecarlo_coups = test_strategie(joueur1, joueur2, nb_bataille)

            plt.hist(montecarlo_coups, bins=20, color='purple', alpha=0.5, label='Stratégie Monte Carlo')
            plt.title("Stratégie Monte Carlo")
            plt.xlabel("Nombre de coups")
            plt.ylabel("Fréquence")
            plt.legend()
            plt.show()
            print(f"Espérance Monte Carlo: {np.mean(montecarlo_coups)}")

        elif choix == '5':
            aleatoire_coups = []
            heuristique_coups = []
            probabiliste_coups = []
            montecarlo_coups = []
            nb_bataille = 100

            # Test JoueurAleatoire
            grille1 = Grille(10)
            grille2 = Grille(10)
            bateaux1 = [Bateau("torpilleur"), Bateau("sous-marin"), Bateau("contre-torpilleurs"), Bateau("croiseur"), Bateau("porte-avions")]
            bateaux2 = [Bateau("torpilleur"), Bateau("sous-marin"), Bateau("contre-torpilleurs"), Bateau("croiseur"), Bateau("porte-avions")]

            joueur1 = JoueurAleatoire(grille1, bateaux1, "Player 1")
            joueur2 = JoueurAleatoire(grille2, bateaux2, "Player 2")

            bataille = Bataille(joueur1, joueur2)
            aleatoire_coups = comparer_strategies(nb_bataille, bataille)


            # Test JoueurHeuristique
            grille1 = Grille(10)
            grille2 = Grille(10)
            bateaux1 = [Bateau("torpilleur"), Bateau("sous-marin"), Bateau("contre-torpilleurs"), Bateau("croiseur"), Bateau("porte-avions")]
            bateaux2 = [Bateau("torpilleur"), Bateau("sous-marin"), Bateau("contre-torpilleurs"), Bateau("croiseur"), Bateau("porte-avions")]
            joueur1 = JoueurHeuristique(grille1, bateaux1, "Player 1")
            joueur2 = JoueurHeuristique(grille2, bateaux2, "Player 2")

            bataille = Bataille(joueur1, joueur2)
            heuristique_coups = comparer_strategies(nb_bataille, bataille)


            # Test JoueurProbabiliste
            grille1 = Grille(10)
            grille2 = Grille(10)
            bateaux1 = [Bateau("torpilleur"), Bateau("sous-marin"), Bateau("contre-torpilleurs"), Bateau("croiseur"), Bateau("porte-avions")]
            bateaux2 = [Bateau("torpilleur"), Bateau("sous-marin"), Bateau("contre-torpilleurs"), Bateau("croiseur"), Bateau("porte-avions")]
            joueur1 = JoueurProbabiliste(grille1, bateaux1, "Player 1")
            joueur2 = JoueurProbabiliste(grille2, bateaux2, "Player 2")

            bataille = Bataille(joueur1, joueur2)
            probabiliste_coups = comparer_strategies(nb_bataille, bataille)


            # Test JoueurMonteCarlo
            grille1 = Grille(10)
            grille2 = Grille(10)
            bateaux1 = [Bateau("torpilleur"), Bateau("sous-marin"), Bateau("contre-torpilleurs"), Bateau("croiseur"), Bateau("porte-avions")]
            bateaux2 = [Bateau("torpilleur"), Bateau("sous-marin"), Bateau("contre-torpilleurs"), Bateau("croiseur"), Bateau("porte-avions")]
            joueur1 = JoueurMonteCarloProbabilisteHeuristique(grille1, bateaux1, "Player 1")
            joueur2 = JoueurMonteCarloProbabilisteHeuristique(grille2, bateaux2, "Player 2")

            bataille = Bataille(joueur1, joueur2)
            montecarlo_coups = comparer_strategies(nb_bataille, bataille)


            # Tracez les résultats
            plt.hist(aleatoire_coups, bins=20, color='blue', alpha=0.5, label='Stratégie aléatoire')
            plt.hist(heuristique_coups, bins=20, color='green', alpha=0.5, label='Stratégie heuristique')
            plt.hist(probabiliste_coups, bins=20, color='red', alpha=0.5, label='Stratégie probabiliste')
            plt.hist(montecarlo_coups, bins=20, color='purple', alpha=0.5, label='Stratégie Monte Carlo')

            plt.title("Comparaison des stratégies (Nombre de coups)")
            plt.xlabel("Nombre de coups")
            plt.ylabel("Fréquence")
            plt.legend()
            plt.show()

            print(f"Espérance aléatoire: {np.mean(aleatoire_coups)}")
            print(f"Espérance heuristique: {np.mean(heuristique_coups)}")
            print(f"Espérance probabiliste: {np.mean(probabiliste_coups)}")
            print(f"Espérance Monte Carlo: {np.mean(montecarlo_coups)}")

        else:
            print("Choix invalide. Veuillez entrer un numéro entre 0 et 5.")

if __name__ == "__main__":
    main()
