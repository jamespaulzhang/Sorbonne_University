# Sam ASLO 21210657
# Yuxiang ZHANG 21202829

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy import stats
import networkx as nx
from collections import defaultdict
from scipy.stats import norm

def get_valid_indices_all_vars(data_list):
    """
    Retourne les indices valides pour chaque colonne dans les données, en ne considérant que les valeurs numériques non manquantes.
    """
    # Initialiser tous les indices comme valides
    valid_indices = np.ones(len(data_list[0]), dtype=bool)
    
    # Vérifier chaque colonne pour s'assurer que les données sont des nombres valides
    for data in data_list:
        valid_indices &= np.isfinite(data) & (np.isreal(data)) 
    
    return np.where(valid_indices)[0]

def compute_LER(d):
    """
    Calculer le LER pour chaque couple de colonnes de données : Crop_1 et Crop_2.
    Renvoie le LER total comme somme des LER pour chaque culture.
    """
    # Récupérer les indices des données valides
    valid_indices = get_valid_indices_all_vars([d['Crop_1_yield_sole'].to_numpy(),
                                                 d['Crop_2_yield_sole'].to_numpy(),
                                                 d['Crop_1_yield_intercropped'].to_numpy(),
                                                 d['Crop_2_yield_intercropped'].to_numpy()])
    
    # Calculer les LER pour chaque culture
    LER_crop1 = d.loc[valid_indices, 'Crop_1_yield_intercropped'] / d.loc[valid_indices, 'Crop_1_yield_sole']
    LER_crop2 = d.loc[valid_indices, 'Crop_2_yield_intercropped'] / d.loc[valid_indices, 'Crop_2_yield_sole']
    LER_tot = LER_crop1 + LER_crop2  # LER total

    return LER_tot

def filter_d2_valid_entries(d):
    """
    Retourne un DataFrame avec seulement les lignes où 'LER_tot' et 'LER_tot_calc' sont tous deux valides.
    """
    valid_indices = get_valid_indices_all_vars([d['LER_tot'].to_numpy(), d['LER_tot_calc'].to_numpy()])
    return d.loc[valid_indices]
    
def plot_LERs(LER_calc, LER_actual):
    """
    Tracer un graphique de dispersion entre les LER calculés et réels, en calculant et en traçant une ligne de régression linéaire, 
    tout en calculant l'erreur quadratique moyenne (RMSE) et le coefficient de détermination (R²).

    Paramètres :
    - LER_calc : valeurs des LER calculés.
    - LER_actual : valeurs des LER réels.

    Retour :
    - RMSE : erreur quadratique moyenne entre les valeurs calculées et réelles.
    - R² : coefficient de détermination (R²) entre les valeurs calculées et réelles.
    """
    # Ajuster le modèle de régression linéaire
    model = LinearRegression()
    LER_calc = np.array(LER_calc).reshape(-1, 1)  # Reshaper les LER calculés pour l'ajustement
    model.fit(LER_calc, LER_actual)
    
    # Prédire les valeurs à partir du modèle de régression
    LER_pred = model.predict(LER_calc)

    # Calcul du RMSE
    rmse = mean_squared_error(LER_actual, LER_pred, squared = True)  # Notez que squared=False pour obtenir la racine carrée

    # Calcul du R² à partir du modèle
    r2 = model.score(LER_calc, LER_actual)  # Utiliser LER_calc pour le score, pas LER_pred
    
    # Tracer le graphique de dispersion
    plt.figure(figsize = (8, 6))
    plt.scatter(LER_calc, LER_actual, color = 'black', label = 'Calculé vs Réel')
    
    # Tracer la ligne de régression linéaire
    plt.plot(LER_calc, LER_pred, color = 'red', linestyle = '--', label = 'Régression Linéaire')

    # Ajouter les étiquettes et le titre
    plt.xlabel('LER Calculé')
    plt.ylabel('LER Réel')
    plt.title('Comparaison des LER Calculés et Réels avec Régression Linéaire')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Afficher le RMSE et le R²
    print('RMSE :', rmse)
    print('R² :', r2)


def compute_mean_std_inter(LER_data):
    """
    Calculer la moyenne, l'écart type (total et non biaisé) et l'intervalle de confiance à 95 % pour les valeurs LER.
    
    Paramètres:
    - LER_data : Tableau numpy des valeurs LER.
    
    Retourne:
    - mLER : Moyenne des valeurs LER.
    - sLER_biased : Écart type biaisé (ddof=0).
    - sLER_unbiased : Écart type non biaisé (ddof=1).
    - variance_LER : Variance des valeurs LER.
    - conf_interval : Intervalle de confiance à 95 % pour la moyenne.
    """
    # Supprimer les valeurs NaN
    LER_data = LER_data[~np.isnan(LER_data)]
    
    # Calcul de la moyenne
    mLER = np.mean(LER_data)
    
    # Calcul des écarts types avec ddof=0 (biaisé) et ddof=1 (non biaisé)
    sLER_biased = np.std(LER_data, ddof=0)  # Écart type biaisé
    print('sLER_biased :', sLER_biased)
    sLER_unbiased = np.std(LER_data, ddof=1)  # Écart type non biaisé
    print('sLER_unbiased :', sLER_unbiased)
    
    # Calcul de la variance
    variance_LER = np.var(LER_data, ddof=0)  # Variance biaisée
    print('variance_LER :', variance_LER)
    
    # Calcul de l'intervalle de confiance à 95 % (avec la distribution normale)
    N = len(LER_data)
    stderr = sLER_biased / np.sqrt(N)  # Calcul de l'erreur standard
    conf_interval = stats.norm.interval(0.95, loc=mLER, scale=stderr)  # Intervalle de confiance avec la normale
    
    return mLER, sLER_biased, conf_interval

def testH0(LER_data):
    """
    Tester l'hypothèse H0 : mu = 1 contre H1 : mu > 1 en utilisant une distribution normale.
    
    Paramètres:
    - LER_data : Tableau numpy des valeurs LER.
    
    Retourne:
    - Valeur p du test.
    """
    # Supprimer les valeurs NaN
    LER_data = LER_data[~np.isnan(LER_data)]
    
    # Utiliser compute_mean_std_inter pour obtenir la moyenne, l'écart type et l'intervalle de confiance
    mLER, sLER_biased, conf_interval = compute_mean_std_inter(LER_data)
    
    # Taille de l'échantillon
    N = len(LER_data)
    
    # Calcul de l'erreur standard
    stderr = sLER_biased / np.sqrt(N)
    
    # Calcul de la statistique z pour H0: mu = 1, H1: mu > 1
    z_stat = (mLER - 1) / stderr
    
    # Calcul de la p-valeur pour un test unilatéral (H0: mu = 1 contre H1: mu > 1)
    p_value_one_sided = 1 - stats.norm.cdf(z_stat)
    
    # Afficher les résultats
    print(f"Statistique z : {z_stat}")
    print(f"p-valeur pour H0 (mu = 1) contre H1 (mu > 1) : {p_value_one_sided}")
    
    return p_value_one_sided
    

def plot_dist_mean(data):
    """
    Étudier la distribution des moyennes pour différents échantillons
    en utilisant une distribution normale.

    Paramètres:
    - data : Tableau numpy des données de LER.
    """
    # Taille de chaque échantillon (par défaut 30)
    sample_size = 30
    # Nombre d'échantillons à générer (par défaut 500)
    num_samples = 500

    # Initialiser un tableau pour stocker les moyennes des échantillons
    sample_means = np.zeros(num_samples)
    
    # Tirer plusieurs échantillons et calculer leur moyenne
    for i in range(num_samples):
        sample = np.random.choice(data, size=sample_size, replace=True)
        sample_means[i] = np.mean(sample)
    
    # Créer un histogramme de la distribution des moyennes des échantillons
    plt.figure(figsize=(10, 6))
    plt.hist(sample_means, bins=100, density=False, color='black', edgecolor='black')
    
    # Ajouter des labels et un titre
    plt.title("Distribution des Moyennes des Échantillons", fontsize=14)
    plt.xlabel("LER", fontsize=12)
    plt.ylabel("Fréquence", fontsize=12)
    
    # Afficher le graphique
    plt.show()
    
def get_sorted_crops(d2):
    """
    Cette fonction calcule la probabilité d'augmentation du rendement pour chaque culture 
    en fonction de son indice LER (Land Equivalent Ratio). Les cultures sont ensuite triées 
    par la probabilité d'augmentation du rendement (probabilité que LER > 1).
    
    Paramètres :
    - d2 : DataFrame contenant les données expérimentales avec les rendements en monoculture 
           et intercalaire des cultures.

    Retourne :
    - Une liste de tuples, chaque tuple contenant le nom de la culture et sa probabilité 
      d'augmentation du rendement, triée par probabilité décroissante.
    """

    # Filtrer les données pour s'assurer qu'il n'y a pas de valeurs manquantes dans les colonnes nécessaires
    valid_data = d2[['Crop_1_Common_Name', 'Crop_2_Common_Name', 
                     'Crop_1_yield_sole', 'Crop_2_yield_sole', 
                     'Crop_1_yield_intercropped', 'Crop_2_yield_intercropped']].dropna()

    # Initialiser un dictionnaire pour stocker les valeurs LER de chaque culture
    crop_LERs = defaultdict(list)

    # Parcourir chaque ligne des données expérimentales, calculer le LER et les enregistrer
    for _, row in valid_data.iterrows():
        # Vérifier s'il n'y a pas de division par zéro, et ignorer ces lignes
        if row['Crop_1_yield_sole'] > 0 and row['Crop_2_yield_sole'] > 0:
            # Calculer le LER pour chaque culture
            LER_crop1 = row['Crop_1_yield_intercropped'] / row['Crop_1_yield_sole']
            LER_crop2 = row['Crop_2_yield_intercropped'] / row['Crop_2_yield_sole']

            # Enregistrer les valeurs LER pour chaque culture
            crop_LERs[row['Crop_1_Common_Name']].append(LER_crop1)
            crop_LERs[row['Crop_2_Common_Name']].append(LER_crop2)

    # Calculer la probabilité d'augmentation du rendement (basée sur la proportion des LER > 1)
    crop_probabilities = []
    for crop, LER_values in crop_LERs.items():
        # Ne considérer que les cultures ayant participé à plus de 10 expériences
        if len(LER_values) >= 10:
            # Assurer que les valeurs LER sont valides, en supprimant les valeurs négatives ou nulles
            LER_values = np.array(LER_values)
            LER_values = LER_values[LER_values > 0]  # Supprimer les valeurs LER inférieures à 0
            if len(LER_values) > 0:
                # Calculer la proportion de valeurs LER > 1
                prob_increase = np.sum(LER_values > 1) / len(LER_values)
                crop_probabilities.append((crop, prob_increase))

    # Trier les cultures par probabilité d'augmentation du rendement, en ordre décroissant
    sorted_crops = sorted(crop_probabilities, key=lambda x: x[1], reverse=True)
    
    # Retourner la liste triée des cultures
    return sorted_crops
    
def list_clusters(d2, th=1.8):
    """
    Détermine les groupes de cultures qui s'associent bien (LER > th). 
    Utilise networkx pour isoler les composantes connexes du graphe.

    Paramètres:
    - d2: DataFrame contenant les données des expériences avec les colonnes 'LER_tot_calc' et les noms de cultures.
    - th: seuil pour le LER total (par défaut = 1.8)

    Retourne:
    - Liste des clusters de cultures qui s'associent bien (LER > th)
    """
    # Créer un graphe vide
    G = nx.Graph()

    # Ajouter des arêtes pour chaque paire de cultures où LER_tot_calc > th
    for _, row in d2.iterrows():
        crop_1 = row['Crop_1_Common_Name']
        crop_2 = row['Crop_2_Common_Name']
        
        # Vérifier si LER_tot_calc dépasse le seuil
        if row['LER_tot_calc'] > th:
            G.add_edge(crop_1, crop_2)

    # Trouver les composantes connexes (clusters)
    clusters = list(nx.connected_components(G))

    # Filtrer et afficher chaque cluster
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx}")
        print("-----------")
        for crop in sorted(cluster):  # Tri alphabétique pour chaque cluster
            print(crop)
        print("-----------")
