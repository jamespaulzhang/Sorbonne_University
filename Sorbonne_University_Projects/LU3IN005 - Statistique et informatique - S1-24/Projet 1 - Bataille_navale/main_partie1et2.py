# Sam ASLO 21210657
# Yuxiang ZHANG 21202829

from typing import Tuple
from grille import Grille
from bateau import Bateau
from functions import *

def demander_coordonnees() -> Tuple[int, int]:
    """
    Demande à l'utilisateur d'entrer des coordonnées valides (x, y) avec 0 <= x <= 9 et 0 <= y <= 9.
    Retourne les coordonnées sous forme de tuple (x, y).
    """
    while True:
        try:
            x = int(input("Entrez la coordonnée x (0-9): "))
            y = int(input("Entrez la coordonnée y (0-9): "))
            if 0 <= x <= 9 and 0 <= y <= 9:
                return x, y
            else:
                print("Les coordonnées doivent être comprises entre 0 et 9. Veuillez réessayer.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer des nombres entiers.")

def choisir_bateau(bateaux) -> Bateau:
    """
    Permet à l'utilisateur de choisir un bateau dans la liste.
    Retourne un objet Bateau correspondant au choix de l'utilisateur.
    """
    print("\n=== Sélectionner un bateau ===")
    for i, bateau in enumerate(bateaux, 1):
        print(f"{i}. {bateau.name} (taille: {bateau.size})")
    
    while True:
        try:
            choix = int(input("Entrez le numéro du bateau que vous souhaitez choisir: "))
            if 1 <= choix <= len(bateaux):
                return bateaux[choix - 1]  # Retourne le bateau correspondant à l'index
            else:
                print(f"Veuillez entrer un numéro entre 1 et {len(bateaux)}.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un numéro.")


def demander_taille_grille() -> int:
    """
    Demande à l'utilisateur d'entrer une taille de grille (entre 5 et 20 par exemple).
    Retourne un entier représentant la taille de la grille.
    """
    while True:
        try:
            taille = int(input("Entrez la taille de la grille (5-20): "))
            if 5 <= taille <= 20:
                return taille
            else:
                print("La taille doit être comprise entre 5 et 20. Veuillez réessayer.")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre entier.")

def main() -> None:
    bateaux = [
        Bateau("porte-avions"), Bateau("croiseur"), 
        Bateau("contre-torpilleurs"), Bateau("sous-marin"), 
        Bateau("torpilleur")
    ]
    grille = None  # Grille principale pour les options 1 à 4
    grille_personnalisee = None  # Grille personnalisée pour les options 5 et 6

    while True:
        print("\n=== Menu Principal ===")
        print("1. Générer une grille 10x10 avec des bateaux aléatoires")
        print("2. Vérifier le placement d'un bateau dans la grille 10x10")
        print("3. Placer un bateau manuellement dans la grille 10x10")
        print("4. Afficher la grille 10x10 actuelle")
        print("5. Calculer les configurations valides pour les bateaux dans une grille personnalisée")
        print("6. Calculer le nombre d'essais pour une grille personnalisée identique")
        print("0. Quitter")

        choix = input("Entrez le numéro de votre choix: ")

        if choix == "1":
            print("Génération de la grille avec des bateaux aléatoires...")
            grille = genere_grille()  # Génère une grille 10x10
            print("Grille générée avec succès.")
            affiche(grille, False)

        elif choix == "2":
            if grille is None:
                print("Veuillez d'abord générer une grille (option 1).")
            else:
                bateau = choisir_bateau(bateaux)  # Permet à l'utilisateur de choisir un bateau
                print(f"Vérification du placement du {bateau.name}...")
                x, y = demander_coordonnees()  # Demander les coordonnées à l'utilisateur
                direction = int(input("Entrez la direction (1 pour verticale, 2 pour horizontale): "))

                if peut_placer(grille, bateau, (x, y), direction):
                    print(f"Le {bateau.name} peut être placé à la position ({x}, {y}) en direction {direction}.")
                else:
                    print(f"Le {bateau.name} ne peut pas être placé à la position ({x}, {y}).")

        elif choix == "3":
            if grille is None:
                print("Veuillez d'abord générer une grille (option 1).")
            else:
                bateau = choisir_bateau(bateaux)  # Permet à l'utilisateur de choisir un bateau
                print(f"Essai de placement du {bateau.name}...")
                x, y = demander_coordonnees()  # Demander les coordonnées à l'utilisateur
                direction = int(input("Entrez la direction (1 pour verticale, 2 pour horizontale): "))

                if place(grille, bateau, (x, y), direction):
                    print(f"Le {bateau.name} a été placé avec succès à la position ({x}, {y}) en direction {direction}.")
                    affiche(grille, False)
                else:
                    print(f"Impossible de placer le {bateau.name} à la position ({x}, {y}).")
                
        elif choix == "4":
            if grille is None:
                print("Veuillez d'abord générer une grille (option 1).")
            else:
                print("Affichage de la grille actuelle :")
                affiche(grille, False)

        elif choix == "5":
            taille_grille = demander_taille_grille()  # Demande à l'utilisateur de définir la taille de la grille
            print(f"Calcul du nombre total de configurations valides pour placer les bateaux dans une grille de taille {taille_grille}x{taille_grille}...")
            grille_personnalisee = Grille(taille_grille)  # Crée la grille personnalisée
            total_configurations = count_configs(grille_personnalisee, bateaux)
            print(f"Nombre total de configurations valides possibles pour placer les bateaux dans une grille de {taille_grille}x{taille_grille} : {total_configurations}")

        elif choix == "6":
            if grille_personnalisee is None:
                print("Veuillez d'abord créer une grille personnalisée dans l'option 5.")
            else:
                print("Calcul du nombre d'essais nécessaires pour générer une grille identique...(il faut attendre un peu)")
                nb_essais = nb_grilles(grille_personnalisee)
                print(f"Nombre d'essais pour générer une grille identique : {nb_essais}")

        elif choix == "0":
            print("Au revoir!")
            break

        else:
            print("Choix non valide, veuillez réessayer.")

if __name__ == "__main__":
    main()