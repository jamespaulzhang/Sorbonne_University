# Yuxiang ZHANG 21202829

import numpy as np
import matplotlib.pyplot as plt

def analyse_rapide(d):
    """
    Analyse rapide d'une variable aléatoire (ici : distance)
    Entrée :
        d : tableau numpy (unidimensionnel, contenant les données d'une variable)
    Sortie :
        Affiche la moyenne, l'écart-type et 10 quantiles
    """

    # Calcul de la moyenne et de l'écart-type
    moyenne = np.mean(d)
    ecart_type = np.std(d)

    # Calcul des 10 quantiles (0%, 10%, 20%, ..., 90%)
    quantiles = np.quantile(d, np.linspace(0.0, 0.9, 10))

    # Affichage des résultats avec formatage des floats
    print(f"mean:{moyenne}")
    print(f"std={ecart_type}")
    
    # Formatage spécifique pour les quantiles
    print("quantiles : [", end="")
    for i, q in enumerate(quantiles):
        if i == len(quantiles) - 1:
            print(f"{q:.1f}", end="")
        else:
            print(f"{q:.1f}, ", end="")
    print("]")

"""
Question: 
Discuter le nombre d'intervalles pour l'histogramme et trouver une valeur satisfaisante

Réponse: 
Le choix du nombre d’intervalles (ou « bins ») dans un histogramme est essentiel : 
s’il y en a trop peu, l’histogramme devient trop lisse et masque des détails ; 
s’il y en a trop, il devient bruité et difficile à interpréter. 

Plusieurs règles empiriques existent :
- Règle de Sturges : k = ⌈log2(n)⌉ + 1
- Règle de la racine carrée : k ≈ √n
- Règle de Rice : k = 2 · n^(1/3)
- Méthodes de Scott ou Freedman–Diaconis : basées sur l’écart-type ou l’IQR

Dans notre cas (n ≈ 6428, pas d’outliers ni de valeurs masquées), 
les méthodes de Rice ou de Freedman–Diaconis donnent une valeur autour de 30–40 intervalles, 
ce qui représente un bon compromis entre lisibilité et précision.

Ainsi, un choix satisfaisant est d’environ 35 intervalles.
"""

def discretisation_histogramme(d, n):
    """
    Discrétisation en n intervalles de largeur constante + histogramme
    Entrée :
        d : tableau numpy unidimensionnel (variable distance)
        n : nombre d'intervalles
    Sortie :
        Affiche les bornes et les effectifs
        Trace l'histogramme
    """

    # 1. Calcul des bornes des intervalles (n intervalles => n+1 bornes)
    bornes = np.linspace(d.min(), d.max(), n+1)

    # 2. Calcul des effectifs par intervalle (manuel)
    effectifs = []
    for i in range(n):
        a, b = bornes[i], bornes[i+1]
        count = np.where((d >= a) & (d <= b), 1, 0).sum()
        effectifs.append(count)
    effectifs = np.array(effectifs)

    # 3. Affichage des résultats
    print("Bornes manuelles:", bornes)
    print("Effectifs manuels:", effectifs)

    # 4. Tracer l'histogramme avec plt.bar
    centres = (bornes[:-1] + bornes[1:]) / 2
    plt.bar(centres, effectifs, width=(bornes[1]-bornes[0]))
    plt.xlabel("Distance")
    plt.ylabel("Effectif")
    plt.title("Histogramme (calculé à la main)")
    plt.show()

    # 5. Vérification avec np.histogram et plt.hist
    effectifs_np, bins_np = np.histogram(d, bins=n)
    print("Effectifs np.histogram:", effectifs_np)

    plt.hist(d, bins=n)
    plt.xlabel("Distance")
    plt.ylabel("Effectif")
    plt.title("Histogramme (np.histogram / plt.hist)")
    plt.show()

def discretisation_prix_au_km(data, n):
    """
    Discrétisation et histogramme du prix au km
    Entrée :
        data : tableau numpy 2D
               suppose que l'avant-dernière colonne est le prix et la dernière colonne est la distance
        n : nombre d'intervalles
    Sortie :
        Affiche les bornes et les effectifs
        Trace l'histogramme
    """

    # 1. Calcul du prix au km
    prix_au_km = data[:, 10] / data[:, -1]

    # 2. Bornes des intervalles
    bornes = np.linspace(prix_au_km.min(), prix_au_km.max(), n+1)

    # 3. Effectifs par intervalle (manuel)
    effectifs = []
    for i in range(n):
        a, b = bornes[i], bornes[i+1]
        count = np.where((prix_au_km >= a) & (prix_au_km < b), 1, 0).sum()
        effectifs.append(count)
    effectifs = np.array(effectifs)

    # 4. Affichage des résultats
    print("Bornes manuelles:", bornes)
    print("Effectifs manuels:", effectifs)

    # 5. Tracer avec plt.bar
    centres = (bornes[:-1] + bornes[1:]) / 2
    plt.bar(centres, effectifs, width=(bornes[1] - bornes[0]))
    plt.xlabel("Prix au km")
    plt.ylabel("Effectifs")
    plt.title(f"Histogramme du prix au km (plt.bar, n={n})")
    plt.show()

    # 6. Vérification avec np.histogram et plt.hist
    effectifs_np, bins_np = np.histogram(prix_au_km, bins=n)
    print("Effectifs np.histogram:", effectifs_np)

    plt.hist(prix_au_km, bins=n)
    plt.xlabel("Prix au km")
    plt.ylabel("Effectifs")
    plt.title(f"Histogramme du prix au km (np.histogram / plt.hist, n={n})")
    plt.show()

"""
Question: 
Proposer un critère rapide pour vérifier que votre distribution conditionnelle respecte bien les propriétés de base

Réponse:  
Un critère simple consiste à vérifier deux points :
Non-négativité : toutes les probabilités doivent être supérieures ou égales à 0.
Normalisation : pour chaque valeur de la variable conditionnante (ici la marque), la somme des probabilités de la variable conditionnée (ici la distance) doit être égale à 1.
En pratique, on peut calculer la somme de chaque distribution conditionnelle par marque et vérifier qu’elle est proche de 1 (à une tolérance numérique près).
"""

"""
Question: 
Cette distribution conditionnelle fait apparaître des pics très marqués : pouvons-nous tirer parti de ces informations?

Réponse:  
Oui. Ces pics indiquent que certaines marques sont fortement associées à des distances particulières. On peut exploiter ces informations de plusieurs façons :
Prédiction : la présence de pics nets permet de mieux prédire la distance typique d’une voiture connaissant sa marque.
Segmentation : on peut distinguer des groupes de marques selon leurs profils de distance, ou fusionner celles qui se ressemblent.
Détection d’anomalies : un trajet dont la distance est très différente du pic attendu pour une marque peut être considéré comme atypique.
"""

def loi_jointe_distance_marque(data, nb_cat, dico_marques, save_pdf=False):
    """
    Calcule et affiche la loi jointe distance/marque (distribution conjointe)
    
    Cette fonction discrétise la variable distance en catégories et calcule 
    la distribution de probabilité conjointe entre les catégories de distance 
    et les marques de véhicules.
    
    Entrées :
        data : tableau numpy 2D contenant les données
        nb_cat : nombre de catégories pour la discrétisation de la distance
        dico_marques : dictionnaire faisant correspondre les noms de marques à leurs indices
        save_pdf : booléen indiquant si la figure doit être sauvegardée en PDF (optionnel)
    
    Sortie :
        p_dm : matrice de probabilités jointes (nb_cat x nb_marques)
                où p_dm[i,j] = P(distance ∈ catégorie i et marque = j)
    
    Étapes du traitement :
    1. Extraction des colonnes distance et marque
    2. Discrétisation de la distance en intervalles égaux
    3. Comptage des co-occurrences avec np.where
    4. Normalisation pour obtenir des probabilités
    5. Visualisation par heatmap et option de sauvegarde
    """

    # 1. Extraction des colonnes pertinentes
    # Dernière colonne = distance, colonne 11 = indices des marques
    d = data[:, -1]        # vecteur des distances
    marques_idx = data[:, 11].astype(int)  # conversion en entiers

    # 2. Discrétisation de la variable distance
    # Création de bornes équidistantes pour nb_cat intervalles
    bins = np.linspace(d.min(), d.max(), nb_cat+1)
    # Attribution de chaque distance à une catégorie (0 à nb_cat-1)
    distance_discretisee = np.digitize(d, bins) - 1
    print("Distance discrétisée:", distance_discretisee)
    
    # 3. Construction de la matrice de comptage conjoint
    # Utilisation de np.where pour compter les occurrences simultanées
    p_dm = np.zeros((nb_cat, len(dico_marques)))
    for dist_cat in range(nb_cat):
        for marque in range(len(dico_marques)):
            # Comptage des individus dans la catégorie de distance et marque données
            count = np.where((distance_discretisee == dist_cat) & (marques_idx == marque), 1, 0).sum()
            p_dm[dist_cat, marque] = count

    # 4. Normalisation pour obtenir une distribution de probabilité
    # La somme de toutes les probabilités jointes vaut 1
    p_dm = p_dm / p_dm.sum()

    # 5. Visualisation de la loi jointe
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # Heatmap avec interpolation nearest pour une visualisation précise
    im = ax.imshow(p_dm, interpolation='nearest', aspect='auto')
    ax.set_xlabel("Marque")
    ax.set_ylabel("Catégories de distance")
    # Configuration des ticks et labels pour l'axe des marques
    ax.set_xticks(np.arange(len(dico_marques)))
    ax.set_xticklabels(dico_marques.keys(), rotation=90, fontsize=8)
    # Barre de couleur indiquant l'échelle des probabilités
    plt.colorbar(im, ax=ax, label="Probabilité jointe")
    plt.title("Distribution conjointe : Distance / Marque")
    plt.tight_layout()
    
    # Option de sauvegarde en format vectoriel (PDF)
    if save_pdf:
        plt.savefig('loi_jointe_distance_marque.pdf', format='pdf', bbox_inches='tight')
        print("Figure sauvegardée sous 'loi_jointe_distance_marque.pdf'")
    
    plt.show()
    return p_dm

def loi_conditionnelle(jointe_dm):
    """
    Calcule la loi conditionnelle P(distance | marque)
    Entrée :
        jointe_dm : matrice (nb_cat x nb_marques) représentant la loi jointe
    Sortie :
        p_d_m : matrice (nb_cat x nb_marques) représentant la loi conditionnelle
    """

    # 1. Calcul de la marginale sur les marques (somme sur les distances)
    marginale_marques = jointe_dm.sum(axis=0)  # taille = nb_marques

    # 2. Calcul de la conditionnelle
    #    Attention à la division par zéro : on garde les colonnes non nulles
    p_d_m = np.zeros_like(jointe_dm)
    for j in range(jointe_dm.shape[1]):
        if marginale_marques[j] > 0:
            p_d_m[:, j] = jointe_dm[:, j] / marginale_marques[j]

    # 3. Affichage
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(p_d_m, interpolation='nearest', aspect='auto')
    plt.colorbar(im, ax=ax)
    plt.title("Loi conditionnelle")
    plt.tight_layout()
    plt.show()
    return p_d_m

def check_conditionnelle(p_d_m, tol=1e-8):
    """
    Vérifie si la matrice est bien une distribution conditionnelle.
    Chaque colonne doit sommer à 1 (ou à 0 si aucune occurrence de la marque).
    """
    col_sums = p_d_m.sum(axis=0)
    return bool(np.all((np.abs(col_sums - 1) < tol) | (col_sums == 0)))

def trace_trajectoires(data):
    """
    Trace toutes les trajectoires (sans distinction de ville, juste segments).
    """
    # Extraire coordonnées départ / arrivée
    x_dep, y_dep = data[:, 6], data[:, 7]
    x_arr, y_arr = data[:, 8], data[:, 9]
    
    # Tracer chaque segment
    for xd, yd, xa, ya in zip(x_dep, y_dep, x_arr, y_arr):
        plt.plot([xd, xa], [yd, ya], color = np.random.rand(3,))

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Trajectoires Blablacar (couleurs aléatoires)")
    plt.show()

def calcule_matrice_distance(data, coord):
    """
    Calcule la matrice des distances entre chaque point de départ (data) et les coordonnées des villes.
    data : tableau avec colonnes (x_dep, y_dep)
    coord : tableau des villes (nb_villes x 2)
    retourne : matrice distances (Ntrajets x Nville)
    """
    x_dep, y_dep = data[:, 6], data[:, 7]   # colonnes départ
    dep_points = np.vstack((x_dep, y_dep)).T  # (Ntrajets x 2)
    # Broadcasting pour calculer distances
    dists = np.linalg.norm(dep_points[:, None, :] - coord[None, :, :], axis=2)
    return dists

def calcule_coord_plus_proche(matrice_dist):
    """
    Trouve l'indice de la ville la plus proche pour chaque trajet
    """
    return np.argmin(matrice_dist, axis=1)

def trace_ville_coord_plus_proche(data, ville_c):
    """
    Trace les trajectoires en fonction de la ville d'origine la plus proche (couleurs)
    """
    x_dep, y_dep = data[:, 6], data[:, 7]
    x_arr, y_arr = data[:, 8], data[:, 9]
    delta_x = x_arr - x_dep
    delta_y = y_arr - y_dep

    plt.quiver(x_dep, y_dep, delta_x, delta_y, color=ville_c,
               angles='xy', scale_units='xy', scale=1)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Trajectoires colorées selon la ville d'origine la plus proche")
    plt.show()

def test_correlation_distance_prix(data):
    """
    Calcul et affichage de la corrélation distance ↔ prix.
    On ignore les trajets où le prix ou la distance est manquant.
    """
    distance = data[:, -1]   # colonne distance
    prix = data[:, 10]       # colonne prix

    # Moyennes
    mean_d = np.mean(distance)
    mean_p = np.mean(prix)

    # Covariance et écarts-types
    cov = np.sum((distance - mean_d) * (prix - mean_p))
    std_d = np.sqrt(np.sum((distance - mean_d)**2))
    std_p = np.sqrt(np.sum((prix - mean_p)**2))

    # Coefficient de corrélation
    r = cov / (std_d * std_p)

    # Scatter pour visualiser
    plt.scatter(distance, prix)
    plt.xlabel("Distance (km)")
    plt.ylabel("Prix (€)")
    plt.title("Corrélation distance / prix")
    plt.show()

    return r

# Corrélation distance/prix : 0.973 → la distance est fortement proportionnelle au prix, relation quasi linéaire.

def test_correlation_distance_confort(data):
    """
    Calcul et affichage de la corrélation distance ↔ étoiles confort.
    Seuls les points avec confort >= 0 sont pris en compte.
    """
    distance = data[:, -1]
    confort = data[:, 12]

    # On ne garde que les trajets avec étoiles renseignées
    mask = confort >= 0
    distance = distance[mask]
    confort = confort[mask]

    # Moyennes
    mean_d = np.mean(distance)
    mean_c = np.mean(confort)

    # Covariance et écarts-types
    cov = np.sum((distance - mean_d) * (confort - mean_c))
    std_d = np.sqrt(np.sum((distance - mean_d)**2))
    std_c = np.sqrt(np.sum((confort - mean_c)**2))

    # Coefficient de corrélation
    r = cov / (std_d * std_c)

    # Scatter pour visualiser
    plt.scatter(distance, confort)
    plt.xlabel("Distance (km)")
    plt.ylabel("Étoiles confort")
    plt.title("Corrélation distance / confort")
    plt.show()

    return r

# Corrélation distance/confort : 0.049 → la distance et le confort sont quasiment indépendants, aucune relation proportionnelle notable.

def calcule_prix_km_seuillée(data, quantile=0.99):
    """
    Calcule le prix/km et applique un seuillage au quantile donné
    """
    prix = data[:, 10]
    distance = data[:, -1]
    prix_km = prix / distance
    seuil = np.quantile(prix_km, quantile)
    prix_km = np.minimum(prix_km, seuil)
    return prix_km

def discretisation(d, nintervalles=30, eps=1e-6):
    """
    Discrétisation générique d'un vecteur 1D en n intervalles de largeur égale.
    Entrées :
        d : tableau numpy 1D (variable à discrétiser)
        nintervalles : nombre d'intervalles
        eps : petite valeur ajoutée au max pour inclure la borne supérieure
    Sortie :
        d_discretise : tableau numpy 1D (catégories entières de 0 à n-1)
    """
    d = np.asarray(d, dtype=float)
    bornes = np.linspace(d.min(), d.max() + eps, nintervalles + 1)
    d_discretise = np.digitize(d, bornes) - 1
    return d_discretise

def loi_jointe(x, y):
    """
    Calcule la loi jointe P(x,y) à partir de deux variables discrètes.
    
    Entrées :
        x : np.array 1D (valeurs discrètes de 0..nx-1)
        y : np.array 1D (valeurs discrètes de 0..ny-1)
    Sortie :
        p_xy : matrice (nx, ny) contenant la loi jointe normalisée
    """
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)

    nx = x.max() + 1
    ny = y.max() + 1

    p_xy = np.zeros((nx, ny))

    for i in range(len(x)):
        p_xy[x[i], y[i]] += 1

    # Normalisation
    p_xy /= p_xy.sum()

    return p_xy

def analyse_ville_horaires_directions(data, coord):
    """
    Analyse des heures de départ et des directions par ville la plus proche.
    -> Affiche les résultats dans le terminal + génère des graphiques.
    """

    noms_villes = ["Grenoble", "Nantes", "Lille", "Strasbourg", "Bordeaux", "Marseille", "Paris"]

    # ---- ÉTAPE 1 : Attribution ville la plus proche ----
    dep_coords = data[:, 6:8]
    mat_dist = np.zeros((len(data), len(coord)))

    for i, city in enumerate(coord):
        mat_dist[:, i] = np.sqrt((dep_coords[:, 0] - city[0])**2 + (dep_coords[:, 1] - city[1])**2)

    ville_plus_proche = np.argmin(mat_dist, axis=1)

    print("\n=== Distribution des trajets par ville la plus proche ===")
    for i, nom in enumerate(noms_villes):
        count = np.sum(ville_plus_proche == i)
        print(f"{nom:12s}: {count:5d} trajets")

    # ---- ÉTAPE 2 : Analyse temporelle ----
    print("\n=== Analyse horaire par ville ===")
    for i, nom in enumerate(noms_villes):
        heures = data[ville_plus_proche == i, 3]  # colonne 3 = heure
        if len(heures) > 0:
            moyenne = np.mean(heures)
            med = np.median(heures)
            matinaux = np.sum(heures < 8) / len(heures) * 100
            print(f"{nom:12s} → heure moyenne: {moyenne:.2f}h, médiane: {med:.0f}h, % départs avant 8h: {matinaux:.1f}%")

            # Graphique
            plt.hist(heures, bins=24, range=(0, 23), alpha=0.7, label=nom)
            plt.xlabel("Heure de départ")
            plt.ylabel("Nombre de trajets")
            plt.title(f"Distribution des heures de départ - {nom}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

    # ---- ÉTAPE 3 : Analyse directionnelle ----
    dx = data[:, 8] - data[:, 6]
    dy = data[:, 9] - data[:, 7]
    directions = []
    for i in range(len(dx)):
        if abs(dx[i]) > abs(dy[i]):
            directions.append('Est' if dx[i] > 0 else 'Ouest')
        else:
            directions.append('Nord' if dy[i] > 0 else 'Sud')
    directions = np.array(directions)

    print("\n=== Directions principales par ville ===")
    for i, nom in enumerate(noms_villes):
        dir_ville = directions[ville_plus_proche == i]
        unique, counts = np.unique(dir_ville, return_counts=True)

        # Impression dans le terminal
        print(f"{nom:12s}: ", end="")
        for u, c in zip(unique, counts):
            pourc = c / len(dir_ville) * 100
            print(f"{u}={c} ({pourc:.1f}%)  ", end="")
        print("")

        # Graphique
        plt.bar(unique, counts, color=['red', 'blue', 'green', 'orange'])
        plt.title(f"Directions des trajets - {nom}")
        plt.ylabel("Nombre de trajets")
        plt.show()

'''
Résultat obtenu :
=== Distribution des trajets par ville la plus proche ===
Grenoble : 1047 trajets
Nantes : 934 trajets
Lille : 920 trajets
Strasbourg : 954 trajets
Bordeaux : 936 trajets
Marseille : 891 trajets
Paris : 746 trajets

=== Analyse horaire par ville ===
Grenoble → heure moyenne : 13.91h, médiane : 15h, % départs avant 8h : 10.1%
Nantes → heure moyenne : 14.97h, médiane : 16h, % départs avant 8h : 7.8%
Lille → heure moyenne : 13.15h, médiane : 14h, % départs avant 8h : 15.4%
Strasbourg → heure moyenne : 13.75h, médiane : 15h, % départs avant 8h : 10.0%
Bordeaux → heure moyenne : 13.79h, médiane : 14h, % départs avant 8h : 9.1%
Marseille → heure moyenne : 14.20h, médiane : 15h, % départs avant 8h : 8.9%
Paris → heure moyenne : 13.86h, médiane : 14h, % départs avant 8h : 11.3%

=== Directions principales par ville ===
Grenoble : Est = 333 (31.8%), Nord = 133 (12.7%), Ouest = 169 (16.1%), Sud = 412 (39.4%)
Nantes : Est = 174 (18.6%), Nord = 293 (31.4%), Ouest = 174 (18.6%), Sud = 293 (31.4%)
Lille : Est = 6 (0.7%), Nord = 264 (28.7%), Ouest = 318 (34.6%), Sud = 332 (36.1%)
Strasbourg : Est = 58 (6.1%), Nord = 58 (6.1%), Ouest = 288 (30.2%), Sud = 550 (57.7%)
Bordeaux : Est = 297 (31.7%), Nord = 346 (37.0%), Ouest = 207 (22.1%), Sud = 86 (9.2%)
Marseille : Est = 385 (43.2%), Nord = 251 (28.2%), Sud = 255 (28.6%)
Paris : Est = 119 (16.0%), Nord = 178 (23.9%), Ouest = 180 (24.1%), Sud = 269 (36.1%)

Question:
Si vous étiez un journaliste en manque de sujet de reportage, quel(s) graphique(s) calculeriez vous à partir de ces données?

Réponse:
Si j’étais journaliste en quête d’un sujet de reportage, je m’appuierais sur les graphiques générés par la fonction :
tme1.analyse_ville_horaires_directions(data, coord)

Cette fonction calcule automatiquement :
7 graphiques de la distribution des heures de départ par ville,
7 graphiques de la direction principale des trajets par ville.

Avec ces représentations, plusieurs angles journalistiques intéressants apparaissent :

Le rythme de vie selon la ville
Exemple : à Lille, 15.4% des départs ont lieu avant 8h du matin, soit beaucoup plus que dans d’autres villes (Paris : 11.3%, Nantes : 7.8%).
Cela permettrait de titrer : “Les Lillois se lèvent plus tôt que les Nantais !”

Les préférences directionnelles locales
À Strasbourg, près de 58% des trajets partent vers le sud, ce qui est beaucoup plus marqué que dans les autres villes.
À Marseille, 43% des trajets sont orientés vers l’est.
Cela pourrait donner un article du type : “Strasbourg regarde vers le sud, Marseille vers l’est : comment les villes dessinent leur mobilité ?”

L’inégalité du volume des trajets
Grenoble domine avec 1047 trajets, alors que Paris en compte seulement 746.
Titre potentiel : “Paris moins mobile que Grenoble ?”

Ainsi, avec seulement 14 graphiques (heures + directions), on obtient des éléments visuels exploitables pour plusieurs articles, en liant la mobilité aux habitudes sociales, aux contraintes géographiques et au rythme de vie urbain.
'''
