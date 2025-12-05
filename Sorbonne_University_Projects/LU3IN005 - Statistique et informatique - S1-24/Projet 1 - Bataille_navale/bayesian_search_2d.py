# Sam ASLO 21210657
# Yuxiang ZHANG 21202829

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def initialize_grid(N: int, distribution_type: str = 'uniform') -> np.ndarray:
    """
    Initialise la distribution a priori de la probabilité dans une grille.
    :param N: Taille de la grille (N x N)
    :param distribution_type: 'uniform', 'corners', 'center', 'edges', 'random'
    :return: Distribution de probabilité π
    """
    grid = np.zeros((N, N))

    if distribution_type == 'uniform':
        grid.fill(1 / (N * N))  # Distribution uniforme
    elif distribution_type == 'corners':
        grid[0, 0] = grid[0, N-1] = grid[N-1, 0] = grid[N-1, N-1] = 1 / 4  # Probabilité uniquement dans les quatre coins
    elif distribution_type == 'center':
        grid[N//2, N//2] = 1  # La probabilité est de 1 au centre, et de 0 ailleurs
    elif distribution_type == 'edges':
        # Distribution sur les bords
        grid[0, :] = grid[N-1, :] = grid[:, 0] = grid[:, N-1] = 1 / (4 * N)  # Probabilité égale le long des bords
    elif distribution_type == 'random':
        grid = np.random.rand(N, N)
        grid /= np.sum(grid)  # Normalisation pour obtenir une distribution de probabilité

    return grid

def detect_object(probability_grid: np.ndarray, detection_success_rate: float) -> Tuple[int, int, int]:
    """
    Simule la recherche dans une cellule et renvoie le résultat.
    :param probability_grid: Grille de probabilité actuelle
    :param detection_success_rate: Taux de succès de la détection
    :return: Résultat de la détection (1 = objet trouvé, 0 = objet non trouvé)
    """
    flattened_grid = probability_grid.flatten()

    # Assure que la distribution de probabilité est valide, évite que la somme soit nulle
    if np.sum(flattened_grid) == 0:
        return 0, 0, 0  # Impossible de détecter l'objet

    # Choisit une cellule aléatoirement
    i, j = np.unravel_index(np.random.choice(flattened_grid.size, p=flattened_grid / np.sum(flattened_grid)), probability_grid.shape)

    # Simule la détection avec une condition pour refléter que l'objet peut ne pas être toujours détecté immédiatement
    found = np.random.rand() < detection_success_rate if np.random.rand() < probability_grid[i, j] else 0

    return found, i, j


def update_probabilities(probability_grid: np.ndarray, detected: int, i: int, j: int, detection_success_rate: float) -> np.ndarray:
    """
    Fonction qui met à jour la probabilité dans la grille.
    :param probability_grid: Grille de probabilité actuelle
    :param detected: Résultat de la détection
    :param i: Ligne de la cellule détectée
    :param j: Colonne de la cellule détectée
    :param detection_success_rate: Taux de succès de la détection
    :return: Grille de probabilité mise à jour
    """
    N = probability_grid.shape[0]
    pi_k = probability_grid[i, j]

    if detected:
        # Si l'objet est trouvé, la probabilité devient 1 pour cette cellule, 0 pour toutes les autres
        probability_grid = np.zeros_like(probability_grid)
        probability_grid[i, j] = 1.0
    else:
        # Met à jour la probabilité de la cellule k
        probability_grid[i, j] = pi_k * (1 - detection_success_rate) / (1 - detection_success_rate * pi_k)

        # Met à jour la probabilité des autres cellules i ≠ k
        for x in range(N):
            for y in range(N):
                if (x != i or y != j):
                    probability_grid[x, y] /= 1 - detection_success_rate * pi_k

    # Normalise la probabilité
    probability_grid /= np.sum(probability_grid)

    return probability_grid

def search_object(N: int, detection_success_rate: float, distribution_type: str = 'uniform', max_iterations: int = 100) -> np.ndarray:
    """
    Effectue le processus de recherche.
    :param N: Taille de la grille
    :param detection_success_rate: Taux de succès de la détection
    :param distribution_type: Type de distribution de probabilité
    :param max_iterations: Nombre maximum d'itérations
    """
    # Initialise la grille
    probability_grid = initialize_grid(N, distribution_type)

    for iteration in range(max_iterations):
        found, i, j = detect_object(probability_grid, detection_success_rate)
        print(f"Iteration {iteration + 1}: Recherche dans la cellule ({i}, {j}) - Détecté: {found}")

        # Met à jour la probabilité
        probability_grid = update_probabilities(probability_grid, found, i, j, detection_success_rate)

        # Affiche la grille de probabilité mise à jour
        print(f"Grille de probabilité mise à jour après l'itération {iteration + 1}:\n", probability_grid)

        if found:
            break  # Si l'objet est trouvé, arrête la recherche

    return probability_grid

def plot_probability_distribution(probability_grid: np.ndarray, title: str) -> None:
    """
    Affiche la distribution de probabilité.
    :param probability_grid: Grille de probabilité
    :param title: Titre du graphique
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(probability_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Probabilité')
    plt.title(title)
    plt.xlabel('Colonne')
    plt.ylabel('Ligne')
    plt.xticks(range(probability_grid.shape[1]))
    plt.yticks(range(probability_grid.shape[0]))
    plt.show()

def main() -> None:
    # Paramètres
    N = 5  # Taille de la grille
    detection_success_rate = 0.7  # Taux de succès de la détection
    max_iterations = 10  # Nombre maximum d'itérations

    # Tester différentes distributions de probabilité
    distributions = ['uniform', 'corners', 'center', 'edges', 'random']

    for distribution in distributions:
        print(f"\nTest de la distribution: {distribution}")

        # Initialiser la grille et lancer la recherche de l'objet
        final_probabilities = search_object(N, detection_success_rate, distribution_type=distribution, max_iterations=max_iterations)

        # Afficher la distribution finale de probabilité après la recherche
        plot_probability_distribution(final_probabilities, f'Distribution de probabilité après la recherche ({distribution})')


if __name__ == "__main__":
    main()
