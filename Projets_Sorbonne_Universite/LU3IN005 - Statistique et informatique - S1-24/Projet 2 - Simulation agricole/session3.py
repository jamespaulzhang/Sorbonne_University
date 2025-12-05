# Yuxiang ZHANG
# Sam ASLO

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation


"""Partie 1 """


def collision(x1: float, y1: float, r1: float, x2: float, y2: float, r2: float):
    """ Vérifie si deux cercles entrent en collision.
    Paramètres :
    x1, y1, r1 : coordonnées et rayon du premier cercle
    x2, y2, r2 : coordonnées et rayon du deuxième cercle

    Retourne :
    True si les cercles entrent en collision, False sinon.
    """
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance < (r1 + r2)


def multi_intercrop(N: int, L: int, rmin: float, rmax: float):
    """ Génère un ensemble de plantes en utilisant une recherche naïve.
    Paramètres :
    N : nombre de plantes à générer
    L : taille du champ
    rmin, rmax : rayon minimum et maximum des plantes

    Retourne :
    La liste des plantes générées.
    """
    plants = []
    nb_essaye = 0
    while len(plants) < N and nb_essaye < 10 * N:
        nb_essaye += 1
        x = np.random.uniform(0, L)
        y = np.random.uniform(0, L)
        r = np.random.uniform(rmin, rmax)
        if can_place_naif(plants, x, y, r, L):
            plant = {'pos': [x, y], 'r': r}
            plants.append(plant)

    return plants


""" Partie figure Q1.2 """


def random_color():
    """ Génère une couleur aléatoire au format RGB.

    Retourne :
    Un tuple représentant une couleur RGB aléatoire.
    """
    return (np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))


def fig_field(plants: list, L: int):
    """ Affiche un graphique représentant les plantes dans le champ.
    Paramètres :
    plants : liste des plantes avec leurs positions et rayons
    L : taille du champ
    """
    fig, ax = plt.subplots()
    for plant in plants:
        x = plant["pos"][0]
        y = plant["pos"][1]
        r = plant["r"]
        circle = plt.Circle((x, y), r, color=random_color())
        ax.add_patch(circle)

    fig.patch.set_facecolor("darkgoldenrod")
    ax.set_facecolor("darkgoldenrod")

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)

    ax.set_aspect('equal', adjustable='box')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)

    plt.show()


""" Partie Naîf """


def can_place_naif(plants: list, x2: float, y2: float, r2: float, L: int):
    """ Vérifie si un cercle peut être placé (en parcourant chaque cercle/plante et en vérifiant la distance entre eux)
        sans entrer en collision avec les autres cercles et sans dépasser les limites.
    Paramètres :
    plants : liste des plantes existantes avec leurs positions et rayons
    x2, y2, r2 : coordonnées et rayon du cercle à tester
    L : taille du champ

    Retourne :
    True si le cercle peut être placé, False sinon.
    """
    if x2 + r2 >= L or y2 + r2 >= L:
        return False

    if x2 - r2 <= 0 or y2 - r2 <= 0:
        return False

    for plant in plants:
        x1 = plant["pos"][0]
        y1 = plant["pos"][1]
        r1 = plant["r"]
        if collision(x1, y1, r1, x2, y2, r2):
            return False
    return True


def monocrop(N: int, L: list, r: float):
    """
    Génère un ensemble de N plantes avec un rayon donné en utilisant une recherche naïve.

    Paramètres :
    N : nombre de plantes à générer
    L : taille du champ (limitant les coordonnées des plantes)
    r : rayon des plantes

    Retourne :
    Le temps écoulé pour l'opération (en secondes)
    La liste des plantes générées, chaque plante étant un dictionnaire avec une position (x, y) et un rayon.
    """
    start = time.time()
    plants = multi_intercrop(N, L, r, r)
    end = time.time()
    return end - start, plants


""" Partie KD """


# Class KD
class Node:
    def __init__(self, point: tuple[float, float], left=None, right=None):
        """ Constructeur de la classe Node. Crée un noeud avec un point et des fils gauche et droit.
        Paramètres :
        point : le point représenté par ce noeud (tuple de coordonnées)
        left : le noeud gauche (par défaut None)
        right : le noeud droit (par défaut None)
        """
        self.point = point
        self.left = left
        self.right = right


class KDTree:
    def __init__(self):
        """ Constructeur de la classe KDTree. Initialise l'arbre avec une racine à None."""
        self.root = None

    def _insert(self, root, point: tuple[float, float], depth: int = 0):
        """ Insère un point dans l'arbre à la position correcte.
        Paramètres :
        root : racine de l'arbre ou sous-arbre
        point : point à insérer (tuple de coordonnées)
        depth : profondeur de l'arbre (par défaut 0, utilisé pour alterner entre
            le sous-arbre gauche et le sous-arbre droit)

        Retourne :
        La racine du sous-arbre mis à jour.
        """
        if root is None:
            return Node(point)

        cd = depth % 2

        if point[cd] < root.point[cd]:
            root.left = self._insert(root.left, point, depth + 1)
        else:
            root.right = self._insert(root.right, point, depth + 1)

        return root

    def insert_point(self, point: tuple[float, float]):
        """ Insère un point dans l'arbre. Si la racine est None, elle crée la racine.
        Paramètres :
        point : point à insérer (tuple de coordonnées)
        """
        if self.root is None:
            self.root = Node(point)
        else:
            self._insert(self.root, point)

    def _distance(self, p1: tuple[float, float], p2: tuple[float, float]):
        """ Calcule la distance Euclidienne entre deux points.
        Paramètres :
        p1 : premier point (tuple de coordonnées)
        p2 : deuxième point (tuple de coordonnées)

        Retourne :
        La distance Euclidienne entre p1 et p2.
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _search_nearest(self, root, point: tuple[float, float], depth: int = 0, best=None):
        """ Recherche le voisin le plus proche d'un point donné dans l'arbre KD.
        Paramètres :
        root : racine de l'arbre ou sous-arbre
        point : point pour lequel chercher le voisin le plus proche
        depth : profondeur de l'arbre (utilisé pour alterner)
        best : meilleur voisin trouvé jusqu'à présent

        Retourne :
        Le noeud correspondant au voisin le plus proche.
        """
        if root is None:
            return best

        if best is None or self._distance(point, root.point) < self._distance(point, best.point):
            best = root

        cd = depth % 2
        next_branch = None
        opposite_branch = None

        if point[cd] < root.point[cd]:
            next_branch = root.left
            opposite_branch = root.right
        else:
            next_branch = root.right
            opposite_branch = root.left

        best = self._search_nearest(next_branch, point, depth + 1, best)

        if abs(point[cd] - root.point[cd]) < self._distance(point, best.point):
            best = self._search_nearest(opposite_branch, point, depth + 1, best)

        return best

    def search_point(self, point: tuple[float, float]):
        """ Fonction auxiliaire qui appelle la méthode `search_nearest` pour rechercher le voisin le plus proche d'un point.
        Cette fonction simplifie l'interface en permettant d'effectuer la recherche à partir de la racine de l'arbre KD.
        Paramètres :
        point : point à rechercher dans l'arbre (tuple de coordonnées)

    Retourne :
    Le noeud correspondant au voisin le plus proche dans l'arbre KDTree.
    """
        return self._search_nearest(self.root, point)


def can_place_kd(x2: float, y2: float, r2: float, L: int, kd_tree):
    """
    Vérifie si un cercle avec un rayon donné peut être placé sans entrer en collision avec les autres cercles.
    On utilise un arbre KD pour la recherche.

    Paramètres :
    x2, y2 : coordonnées du point à tester
    r2 : rayon du cercle à tester
    L : taille du champ
    kd_tree : l'arbre KDTree utilisé pour trouver les voisins

    Retourne :
    True si le cercle peut être placé sans collision et dans les limites, False sinon.
    """
    # Vérifie si le point est en dehors des limites
    if x2 + r2 >= L or y2 + r2 >= L or x2 - r2 <= 0 or y2 - r2 <= 0:
        return False

    # Recherche le voisin le plus proche dans l'arbre KDTree
    nearest = kd_tree.search_point((x2, y2))

    if nearest is not None:
        x1, y1 = nearest.point
        if collision(x1, y1, r2, x2, y2, r2):
            return False

    return True


def monocrop_KD(N: int, L: int, r: float):
    """
    Génère N cercles/plantes valides dans le champ en utilisant un KDTree pour vérifier les collisions.

    Paramètres :
    N : nombre de points à générer
    L : taille du champ
    r : rayon des points à insérer

    Retourne :
    Le temps écoulé pour l'opération (en secondes)
    La liste des cercles générés, chaque cercle étant un dictionnaire avec une position (x, y) et un rayon.
    """
    start = time.time()

    kd_tree = KDTree()
    plants_KD = []
    nb_essaye = 0

    while len(plants_KD) < N and nb_essaye < 10 * N:
        x = np.random.uniform(0, L)
        y = np.random.uniform(0, L)
        if can_place_kd(x, y, r, L, kd_tree):
            valid_point = {'pos': [x, y], 'r': r}
            plants_KD.append(valid_point)
            kd_tree.insert_point((x, y))

    end = time.time()

    return end - start, plants_KD


""" Partie 2.1"""


# Partie Simulation


# Classe Plante, nous avons décidé de faire cette classe pour rendre le code plus simple
class Plant:
    def __init__(self, x, y, planting_time):
        """
        Constructeur de la classe Plant.
        Initialise une plante avec ses coordonnées (x, y), le temps de plantation,
        le rayon initial et une couleur aléatoire.

        Paramètres :
        x : float, position en x de la plante dans le champ
        y : float, position en y de la plante dans le champ
        planting_time : int, temps auquel la plante a été plantée
        """
        self.x = x
        self.y = y
        self.planting_time = planting_time
        self.r = 0.1
        self.color = random_color()

    def update_r(self, current_time, alpha, th, plants_class):
        """
        Met à jour le rayon de la plante en fonction du temps écoulé depuis la plantation.

        Si le temps actuel est inférieur à th,
        le rayon de la plante augmente de façon linéaire.

        Si la plante a atteint ou dépassé th, elle est retirée de la simulation.

        Paramètres :
        current_time : int, temps actuel de la simulation
        alpha : float, facteur de croissance du rayon
        th : float, seuil de temps après lequel la plante ne grandit plus
        plants_class : list, liste des objets de type Plant dans la simulation
        """
        if current_time - self.planting_time < th:
            new_r = alpha * (current_time - self.planting_time)   # Calcul du nouveau rayon
            # On met à jour le rayon si la nouvelle valeur est plus grande (la première valeur sera inférieure alpha * 0)
            if new_r > self.r:
                self.r = new_r
        else:
            plants_class.remove(self)
            del self


def fig_field2(plants: list[Plant], L: int):
    """ Affiche un graphique représentant les plantes dans le champ.
        Similaire à fig_field mais plants ici est une liste d'objets
    Paramètres :
    plants : liste des plantes avec leurs positions, rayons et couleurs
    L : taille du champ
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)

    patches = []
    for plant in plants:
        patch = plt.Circle((plant.x, plant.y), plant.r, color=plant.color)
        ax.add_patch(patch)
        patches.append(patch)

    fig.patch.set_facecolor("darkgoldenrod")
    ax.set_facecolor("darkgoldenrod")

    ax.set_aspect('equal', adjustable='box')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    plt.show()


def dynamic_random_planting(planting_rate, Rmax, th):
    """
    Fonction qui simule la plantation dynamique de plantes sur une période de temps (th * 4).
    Utilise un modèle de distribution de Poisson pour déterminer le nombre de plantes à chaque instant.

    Paramètres :
    planting_rate : taux de plantation (nombre moyen de plantes par unité de temps)
    Rmax : rayon maximum que peut atteindre une plante
    th : seuil de temps après lequel le rayon des plantes ne grandit plus

    Retourne :
    history : Liste des états des plantes (position, rayon, couleur) à chaque instant
    ps : Liste des informations sur les plantes (position et temps de plantation)
    """
    total_time = th * 4
    planting = np.random.poisson(lam=planting_rate, size=int(total_time))
    L = 100

    plants = []
    ps = []
    plants_class = []
    history = []  # Historique des états des plantes pour chaque itération

    nb_essaies = 10  # Nombre d'essais pour trouver une position valide pour une plante
    alpha = Rmax / th  # Facteur de croissance du rayon par unité de temps
    max_r = alpha * th  # Rayon maximum qu'une plante peut atteindre

    for t in range(len(planting)):
        for j in range(planting[t]):
            for essai in range(nb_essaies):  # Essayer plusieurs fois si nécessaire
                x = np.random.uniform(0, L)
                y = np.random.uniform(0, L)

                if can_place_naif(plants, x, y, max_r, L):
                    plant = {'pos': [x, y], 'r': max_r}
                    plants.append(plant)

                    plant = {'pos': [x, y], 't': t}
                    ps.append(plant)

                    plants_class.append(Plant(x, y, t))
                    break

        for plant in plants_class:
            plant.update_r(t, alpha, th, plants_class)

        # Si pas commenté, on affichera le graphique à chaque itération
        # fig_field2(plants_class, L)

        # Sauvegarde de l'état des plantes pour chaque itération
        history_temp = []
        for plant in plants_class:
            history_temp.append((plant.x, plant.y, plant.r, plant.color))

        history.append(history_temp)

    return history, ps


def fig_dynamic(plants, photo_name, video_name):
    """
    Fonction qui génère des images représentant l'évolution dynamique des plantes et les sauvegarde
    dans un dossier temporaire. Ensuite, elle génère une vidéo à partir de ces images,
    les supprimez tous sauf un au hasard.

    Paramètres :
    plants : Liste des états des plantes à chaque itération
    photo_name : Nom du fichier pour sauvegarder une photo aléatoire de l'évolution
    video_name : Nom du fichier vidéo de sortie
    """
    L = 100
    total_photos = len(plants)

    folder_name = "photos_temp_plot"  # Dossier pour stocker les photos temporaires
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        # Le dossier temporaire sera créé, puis supprimé à la fin de la fonction.
        # Si, par hasard, vous possèdez déjà un dossier portant le même nom,
        # une erreur est levée pour éviter de supprimer ce dossier existant.
        raise FileExistsError("La fonction ne peut pas continuer car un dossier avec le même nom existe déjà")

    # Générer les photos à chaque étape de la simulation
    for photo in range(total_photos):
        file_name = f"plot_{photo:03d}.png"
        fig_field3(plants[photo], L, folder_name, file_name)

    # Créer une vidéo à partir des images sauvegardées
    create_video_from_photos(folder_name, video_name, 2)

    # Enregistrer un snapshot aléatoire d'une photo
    lower_bound = int(total_photos * (3 / 4))  # Choisir une photo aléatoire parmi les dernières étapes
    random_number = np.random.randint(lower_bound, total_photos)
    src_file = f"photos_temp_plot/plot_{random_number:03d}.png"
    current_dir = os.getcwd()
    dest_file = os.path.join(current_dir, photo_name)
    shutil.move(src_file, dest_file)

    # Supprimer le dossier temporaire contenant les photos
    shutil.rmtree(folder_name)


def fig_field3(plants, L, folder_name, file_name):
    """
    Génére un champ avec les plantes et sauvegarde une image de ce champ dans un fichier.

    Paramètres :
    plants : Liste des plantes (position, rayon, couleur)
    L : Taille du champ
    folder_name : Nom du dossier où sauvegarder l'image
    file_name : Nom du fichier image
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)

    patches = []
    for plant in plants:
        patch = plt.Circle((plant[0], plant[1]), plant[2], color=plant[3])
        ax.add_patch(patch)
        patches.append(patch)

    fig.patch.set_facecolor("darkgoldenrod")
    ax.set_facecolor("darkgoldenrod")

    ax.set_aspect('equal', adjustable='box')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    file_path = os.path.join(folder_name, file_name)
    plt.savefig(file_path)
    plt.close(fig)


def create_video_from_photos(image_folder: str, output_video: str, frame_rate: int):
    """
    Crée une vidéo à partir d'un dossier contenant des images PNG.
    Chaque image du dossier est affichée successivement dans la vidéo,

    Paramètres :
    image_folder : chemin vers le dossier contenant les images PNG
    output_video : nom du fichier vidéo de sortie
    frame_rate : taux de rafraîchissement de la vidéo (nombre d'images par seconde)

    Description :
    - La fonction liste toutes les images PNG du dossier spécifié.
    - Les images sont triées par ordre numérique pour garantir un ordre d'affichage correct.
    - Une animation est créée où chaque image est affichée à son tour,
      avec un intervalle déterminé par le taux de rafraîchissement.
    - La vidéo résultante est enregistrée sous le nom de fichier spécifié.
    """
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()

    fig, ax = plt.subplots()

    def update(frame):
        img_path = os.path.join(image_folder, images[frame])
        img = plt.imread(img_path)
        ax.clear()
        ax.imshow(img)
        ax.axis('off')

    ani = animation.FuncAnimation(fig, update, frames=len(images), interval=1000 / frame_rate)
    ani.save(output_video, writer='ffmpeg', fps=frame_rate)
    plt.close(fig)


""" Partie 2.2 """


def dynamic_random_planting_KD(planting_rate, Rmax, th):
    """
    Fonction similaire à dynamic_random_planting, mais utilisant des arbres KD pour
    effectuer les recherches. ce qui rendra la simulation plus rapide surtout si nous
    augmentons le taux de plantation (planting_rate)

    Paramètres :
    planting_rate : taux de plantation (nombre moyen de plantes à planter par unité de temps)
    Rmax : rayon maximum qu'une plante peut atteindre
    th : seuil de temps après lequel le rayon des plantes ne croît plus

    """
    total_time = th * 4
    planting = np.random.poisson(lam=planting_rate, size=int(total_time))
    L = 100

    ps = []
    plants_class = []
    history = []
    kd_tree = KDTree()

    nb_essaies = 10
    alpha = Rmax / th
    max_r = alpha * th

    for t in range(len(planting)):
        for j in range(planting[t]):
            for essai in range(nb_essaies):
                x = np.random.uniform(0, L)
                y = np.random.uniform(0, L)

                """
                Au lieu de parcourir toutes les plantes pour vérifier
                si une nouvelle plante peut être placée sans collison,
                on utilise un arbre KD pour effectuer cette recherche plus rapidement.
                """
                if can_place_kd(x, y, max_r, L, kd_tree):
                    kd_tree.insert_point((x, y))

                    plant = {'pos': [x, y], 't': t}
                    ps.append(plant)

                    plants_class.append(Plant(x, y, t))
                    break

        for plant in plants_class:
            plant.update_r(t, alpha, th, plants_class)

        # Si pas commenté, on affichera le graphique à chaque itération
        fig_field2(plants_class, L)

        history_temp = []
        for plant in plants_class:
            history_temp.append((plant.x, plant.y, plant.r, plant.color))

        history.append(history_temp)

    return history, ps
