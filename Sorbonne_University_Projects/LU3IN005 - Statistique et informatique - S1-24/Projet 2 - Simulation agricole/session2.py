# Sam ASLO 21210657
# Yuxiang ZHANG 21202829

import pandas as pd
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def get_cropping_cycles_database(filepath):
    """
    Charger la base de données des cycles de cultures à partir d'un fichier CSV.

    Paramètres:
    - filepath : Chemin du fichier CSV contenant les informations des cycles de cultures.

    Retour:
    - cropping_cycles_df : DataFrame contenant les données des cycles de cultures.
    """
    cropping_cycles_df = pd.read_csv(filepath)
    return cropping_cycles_df

def get_criteria(filepath):
    """
    Charger les critères de sélection des cultures à partir d'un fichier CSV.

    Paramètres:
    - filepath : Chemin du fichier CSV contenant les critères de sélection.

    Retour:
    - criteria_df : DataFrame contenant les critères pour les cultures.
    """
    criteria_df = pd.read_csv(filepath)
    return criteria_df

def choose_cycle(category, month, cropping_cycles, criteria):
    """
    Sélectionner un cycle de culture pour une catégorie donnée en fonction du mois et des critères.

    Paramètres:
    - category : Catégorie de culture à sélectionner.
    - month : Mois de vente (de 0 à 11 pour janvier à décembre).
    - cropping_cycles : DataFrame des cycles de cultures disponibles.
    - criteria : DataFrame des critères pour déterminer les cultures nécessaires.

    Retour:
    - cycle_id : ID du cycle sélectionné, ou False si aucun cycle ne répond aux critères.
    """
    month_column = f'Minimal number of crops_{month+1}'  # Les mois commencent de 0
    min_crops_needed = criteria.loc[criteria['Crop category'] == category, month_column].values
    
    if min_crops_needed.size == 0 or min_crops_needed[0] == 0:
        return False
    
    sale_column = f'Sale_{["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month]}'
    available_cycles = cropping_cycles[
        (cropping_cycles['Crop category'] == category) &
        (cropping_cycles[sale_column] == 1)
    ]
    
    if available_cycles.empty:
        return False
    
    cycle_id = random.choice(available_cycles['ID'].values)
    return cycle_id

def update_shares(cycle, cycles, cropping_cycles):
    """
    Mettre à jour le nombre de lots (parts) pour un cycle donné.

    Paramètres:
    - cycle : DataFrame contenant les informations du cycle actuel.
    - cycles : Dictionnaire contenant les quantités actuelles de lots par cycle.
    - cropping_cycles : DataFrame des cycles de cultures.

    Retour:
    - cycles : Dictionnaire mis à jour avec les quantités de lots pour le cycle.
    """
    cycle_id = cycle["ID"].values[0]
    shmin = cycle["Shmin"].values[0]
    shmax = cycle["Shmax"].values[0]
    
    if cycle_id in cycles:
        if cycles[cycle_id] < shmax:
            cycles[cycle_id] += 1
    else:
        cycles[cycle_id] = shmin
    
    return cycles

def spread_shares(cycle, cycles, cropping_cycles):
    """
    Répartir les parts (lots) d'un cycle de culture sur les mois disponibles pour la vente.

    Paramètres:
    - cycle : DataFrame contenant les informations du cycle actuel.
    - cycles : Dictionnaire contenant les quantités actuelles de lots par cycle.
    - cropping_cycles : DataFrame des cycles de cultures.

    Retour:
    - monthly_shares : Liste de 12 valeurs représentant les parts mensuelles réparties.
    """
    # Obtenir l'ID du cycle actuel
    cycle_id = int(cycle["ID"].to_numpy()[0])

    # Chercher les données pour ce cycle
    cycle_data = cropping_cycles[cropping_cycles['ID'] == cycle_id]
    if cycle_data.empty:
        raise ValueError(f"Cycle avec l'ID {cycle_id} non trouvé dans les données de cropping_cycles.")
        
    # Obtenir la quantité pour ce cycle
    lot_quantity = cycles.get(cycle_id, 0)

    # Si lot_quantity est une liste (cas peu fréquent), on effectue la somme
    if isinstance(lot_quantity, list):
        lot_quantity = sum(lot_quantity)  # Somme des quantités

    # Obtenir les informations sur les ventes mensuelles
    sale_columns = [f"Sale_{month}" for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]]
    sale_data = cycle[sale_columns].values[0]
    
    # Déterminer les mois avec des ventes
    months_with_sales = [i for i, sale in enumerate(sale_data) if sale == 1]
    
    if not months_with_sales:
        return [np.float32(0)] * 12  # Si aucun mois n'a de vente, retourner une liste de zéros
    
    # Calculer la part par mois
    shares_per_month = lot_quantity / len(months_with_sales)
    monthly_shares = [np.float32(0)] * 12
    
    # Répartir les parts sur les mois
    for month in months_with_sales:
        monthly_shares[month] = np.float32(shares_per_month)
    
    return monthly_shares

def check_quant(monthly_shares, criteria, month):
    """
    Vérifier si la quantité de parts vendues pour un mois donné répond aux exigences minimales
    pour une catégorie donnée, en utilisant les cycles de cultures et les critères.

    Paramètres:
    - monthly_shares : Dictionnaire des parts mensuelles vendues pour chaque cycle.
    - criteria : DataFrame des critères pour une catégorie spécifique.
    - month : Mois en question (de 0 à 11 pour janvier à décembre).

    Retour:
    - bool : True si la quantité minimum est atteinte, sinon False.
    """
    cropping_cycles = get_cropping_cycles_database("data/session2/cropping_cycles.csv")

    # 1. Récupérer les critères de vente minimum pour le mois donné et la catégorie
    min_quantity = criteria[f'Minimal quantity of shares_{month+1}'].values[0]

    # 2. Choisir un cycle de culture basé sur la catégorie et le mois
    cycle_id = choose_cycle(criteria['Crop category'].iloc[0], month, cropping_cycles, criteria)
    if cycle_id is False:
        return False  # Aucun cycle sélectionné, donc ne satisfait pas le critère.

    # 3. Mettre à jour les parts disponibles pour ce cycle
    cycle_data = cropping_cycles[cropping_cycles['ID'] == cycle_id]
    updated_monthly_shares = spread_shares(cycle_data, monthly_shares, cropping_cycles)

    # 4. Vérifier si la quantité pour le mois spécifié est suffisante
    sales_quantity = updated_monthly_shares[month]
    return sales_quantity >= min_quantity
    
def check_div(monthly_shares, cropping_cycles, criteria):
    """
    Vérifie la diversité des cultures pour chaque mois selon les critères.

    Paramètres:
    - monthly_shares : Liste des parts mensuelles vendues pour chaque cycle (12 valeurs par cycle).
    - cropping_cycles : DataFrame des cycles de cultures.
    - criteria : DataFrame contenant les critères de diversité mensuels.

    Retour:
    - bool : True si la diversité minimale est respectée pour chaque mois, sinon False.
    """
    # Parcourt chaque mois (de 0 à 11, correspondant aux 12 mois)
    for month in range(12):
        # Récupère le nombre minimum de cultures nécessaires pour ce mois
        min_crops_needed = criteria[f'Minimal number of crops_{month+1}'].values[0]
        
        # Ensemble des cultures vendues ce mois-ci
        sales_crops = set()

        # Parcourt tous les cycles de culture
        for index, cycle in cropping_cycles.iterrows():
            # Vérifie si ce cycle est vendu pendant ce mois
            if cycle[f'Sale_{["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month]}'] == 1:
                # Sélectionne le cycle correspondant
                cycle_id = choose_cycle(criteria['Crop category'].iloc[0], month, cropping_cycles, criteria)
                if not cycle_id:
                    return False  # Aucun cycle sélectionné, ne satisfait pas le critère
                    
                cycle_data = cropping_cycles[cropping_cycles['ID'] == cycle_id]
                monthly_shares_for_cycle = spread_shares(cycle_data, monthly_shares, cropping_cycles)
                
                # Ajoute le type de culture de ce cycle à l'ensemble des cultures vendues
                sales_crops.add(cycle['Crop_french'])

        # Vérifie si le nombre de cultures vendues respecte le critère minimum pour ce mois
        if len(sales_crops) < min_crops_needed:
            return False  # Si un mois ne respecte pas, retourne False
    
    # Si tous les mois respectent les critères, retourne True
    return True

def get_box_cat(category, criteria, cropping_cycles):
    """
    Génère un panier de cycles de culture pour une catégorie donnée, respectant les contraintes
    de diversité et de quantité.

    Paramètres:
    - category : Catégorie de culture
    - criteria : DataFrame contenant les critères de diversité et de quantité pour chaque mois.
    - cropping_cycles : DataFrame des cycles de cultures disponibles.

    Retour:
    - dict : Dictionnaire avec l'ID du cycle comme clé et le nombre de lots par an comme valeur.
    """
    # Filtre les cycles de culture et les critères pour la catégorie donnée
    criteria_cat = criteria[criteria["Crop category"] == category]
    cycles_cat = cropping_cycles[cropping_cycles["Crop category"] == category]

    box = {}

    # Parcourt chaque mois et ajoute des cycles respectant les critères de diversité et quantité
    for month in range(12):
        required_crops = criteria_cat[f"Minimal number of crops_{month + 1}"].values[0]
        required_quantity = criteria_cat[f"Minimal quantity of shares_{month + 1}"].values[0]
        
        selected_crops = set()
        
        # S'assure que les critères de diversité et de quantité sont respectés
        while required_crops > 0 or required_quantity > 0:
            chosen_cycle = random.choice(cycles_cat["ID"].values)
            
            # Récupère les informations du cycle choisi
            cycle_info = cycles_cat[cycles_cat["ID"] == chosen_cycle]
            shmin = cycle_info["Shmin"].values[0]
            shmax = cycle_info["Shmax"].values[0]

            # Si le cycle est déjà dans le panier, on augmente sa quantité si ce n'est pas au maximum
            if chosen_cycle in box:
                if box[chosen_cycle] < shmax:
                    box[chosen_cycle] += 1
            else:
                box[chosen_cycle] = shmin  # Utilise le minimum pour la première fois
            
            # Met à jour les exigences de diversité
            if chosen_cycle not in selected_crops:
                selected_crops.add(chosen_cycle)
                required_crops -= 1  # Réduit le nombre de cultures nécessaires

            # Met à jour les exigences de quantité
            required_quantity -= 1
            
            # Évite une boucle infinie si les critères ne peuvent pas être satisfaits
            if required_crops <= 0 and required_quantity <= 0:
                break
    return box

def get_N_boxes(N, criteria, cropping_cycles, categories):
    """
    Génère N paniers de cycles de culture en respectant les critères.

    Paramètres:
    - N : Nombre de paniers à générer.
    - criteria : DataFrame des critères de diversité et de quantité.
    - cropping_cycles : DataFrame des cycles de cultures disponibles.
    - categories : Liste des catégories de cultures à prendre en compte.

    Retour:
    - List : Liste contenant les paniers générés.
    """
    all_boxes = []
    
    # Génère N paniers
    for _ in range(N):
        box = {}
        
        # Pour chaque catégorie, génère un panier
        for category in categories:
            box[category] = get_box_cat(category, criteria, cropping_cycles)
        
        all_boxes.append(box)
    
    return all_boxes

def save_boxes_to_csv(boxes, filename):
    """
    Sauvegarde les paniers générés dans un fichier CSV.
    
    Paramètres:
    - boxes : Liste contenant les paniers générés (chaque panier est un dictionnaire).
    - filename : Nom du fichier CSV où les paniers seront enregistrés.
    """
    # Convertit le dictionnaire de paniers en DataFrame
    data = []
    for i, box in enumerate(boxes):
        for category, category_boxes in box.items():
            for cycle_id, quantity in category_boxes.items():
                data.append([category, cycle_id, quantity, i + 1])

    # Crée un DataFrame à partir des données
    df = pd.DataFrame(data, columns=["Category", "Cycle ID", "Quantity", "Basket ID"])
    
    # Sauvegarde en CSV
    df.to_csv(filename, index=False)

english_to_french_crop_names = {
    "Carrot": ["Carotte botte", "Carotte conservation"],
    "Condiment crop": ["Aromatique"],
    "Cooked green": ["Epinard"],
    "Fruit crop": ["Melon", "Fraise"],
    "Potato": ["Pomme de terre conservation", "Pomme de terre primeur"],
    "Raw green": ["Salade", "Mache-pourp", "Mesclun"],
    "Root crop": ["Betterave botte", "Navet botte", "Radis botte"],
    "Tomato": ["Tomate classique", "Tomate cerise", "Tomate ancienne"],
}

def compute_CA_workload(farm, crop_properties):
    """
    Calcule le travail total et le revenu total des cultures configurées,
    ainsi que la surface correspondant à la charge de travail totale.
    
    Paramètres:
    - farm : Dictionnaire contenant les cycles de cultures avec leurs quantités.
    - crop_properties : DataFrame contenant les propriétés des cultures, y compris le prix, la quantité et le travail requis.
    
    Retour:
    - total_workload : Charge de travail totale en heures
    - total_revenue : Revenu total en euros
    """
    
    total_workload = 0
    total_revenue = 0

    # Parcourt chaque cycle de culture dans la ferme
    for category, crop_cycles in farm.items():
        # Traduit les noms de cultures si nécessaire
        french_names = english_to_french_crop_names.get(category, [category])
        crop_info = crop_properties[crop_properties["Crop_french"].isin(french_names)]
        
        if crop_info.empty:
            print(f"Avertissement : Culture '{category}' non trouvée dans crop_properties.")
            continue

        # Récupère les informations sur le prix, la quantité et les effets
        price_per_kg = crop_info["Price"].values[0]
        quantity_per_share = crop_info["Quantity_per_share"].values[0]
        a_c = crop_info["Effect_on Log_Yield"].values[0]
        b_c = crop_info["Effect_on_Log_Production_workload"].values[0]

        # Génère des effets aléatoires pour le calcul
        a_s = np.random.normal(0.74, 0.12)
        a_f = np.random.normal(0, 0.42)
        r = np.random.normal(0, 0.14)
        
        b_s = np.random.normal(2.72, 0.19)
        b_f = np.random.normal(0, 0.36)
        s = np.random.normal(0, 0.21)

        # Calcule le rendement (logarithmique) et la charge de travail (logarithmique)
        log_Y_c = a_s + a_f + a_c + r
        log_W_c = b_s + b_f + b_c + s

        yield_per_m2 = np.exp(log_Y_c)  # Rendement en lots par m²
        workload_per_m2 = np.exp(log_W_c)  # Charge de travail en heures par m²

        # Parcourt chaque cycle de culture dans crop_cycles
        for cycle_id, quantity in crop_cycles.items():
            # Calcule le revenu total pour cette culture
            revenue = price_per_kg * quantity_per_share * yield_per_m2 * quantity
            total_revenue += revenue
            
            # Calcule la charge de travail totale pour cette culture
            total_workload += workload_per_m2 * quantity

    return total_workload, total_revenue

def compute_CA(workloads, CAs, workload_total):
    """
    Calcule le chiffre d'affaires total pour chaque ferme en fonction de la charge de travail donnée,
    et retourne un tableau des résultats.

    Paramètres :
    - workloads : Liste des charges de travail (en heures par m²) pour chaque ferme.
    - CAs : Liste des chiffres d'affaires (en euros par m²) pour chaque ferme.
    - workload_total : Charge de travail totale disponible (par exemple, 1800 heures).

    Retour :
    - CA_totals : Tableau des chiffres d'affaires totaux pour chaque ferme en fonction de la charge de travail donnée.
    """
    
    # Conversion des listes en tableaux numpy pour des calculs efficaces
    workloads = np.array(workloads)
    CAs = np.array(CAs)
    
    # Calcul de la surface nécessaire pour chaque ferme
    required_area = workload_total / workloads
    
    # Calcul du chiffre d'affaires total pour chaque ferme
    CA_totals = required_area * CAs
    
    return CA_totals
    
def compute_probability_viable(boxes, CA_min, workload_max, crop_properties):
    """
    Calcule la probabilité qu'une microferme fasse un chiffre d'affaire de plus de CA_min
    avec une charge de travail inférieure à workload_max.
    
    Paramètres:
    - boxes : Liste des boîtes contenant les informations sur les cultures
    - CA_min : Le chiffre d'affaire minimal requis (en euros)
    - workload_max : La charge de travail maximale autorisée (en heures)
    - crop_properties : Propriétés des cultures contenant des informations sur le travail et le revenu
    
    Retour:
    - proba_viable : La probabilité qu'une microferme soit viable (répond aux critères de travail et de revenu)
    """
    viable_count = 0  # Nombre de fermes viables
    total_count = len(boxes)  # Nombre total de fermes
    
    # Parcours de chaque boîte pour vérifier les conditions de travail et de chiffre d'affaire
    for box in boxes:
        # Calculer le travail total et le chiffre d'affaire total pour cette ferme
        workload, CA = compute_CA_workload(box, crop_properties)
        
        # Vérifier si cette ferme répond aux critères
        if workload <= workload_max and CA >= CA_min:
            viable_count += 1  # Incrémente le nombre de fermes viables
    
    # Calculer la probabilité que la ferme soit viable
    proba_viable = viable_count / total_count if total_count > 0 else 0  # Eviter la division par zéro
    
    return proba_viable


def figure_distribution(ws, CAs):
    """
    Trace un histogramme 2D de la distribution du travail et du profit, où la couleur représente la densité.

    Paramètres:
    ws -- Liste ou tableau de données de charge de travail
    CAs -- Liste ou tableau de données de profit
    """
    plt.figure(figsize=(10, 6))
    
    # Trace la carte de densité avec un histogramme 2D
    plt.hist2d(ws, CAs, bins=100, cmap='plasma')
    
    # Ajoute une barre de couleurs
    plt.colorbar(label='Densité')
    
    # Définit les labels des axes et le titre
    plt.xlabel('Charge de travail')
    plt.ylabel('Profit')
    plt.title('Distribution de la densité entre Profit et Charge de travail')
    
    # Affiche l'image
    plt.show()


def figure_CAtot(CA_total):
    """
    Trace la distribution du chiffre d'affaires total (CA_total).

    Paramètres:
    CA_total -- Liste ou tableau de données du chiffre d'affaires total
    """
    plt.figure(figsize=(8, 6))
    
    # Trace la distribution des données avec un histogramme
    plt.hist(CA_total, bins=100, color='black')
    
    # Définit les labels des axes X et Y, ainsi que le titre
    plt.xlabel("Chiffre d'affaires")
    plt.ylabel("Fréquence")
    plt.title("Distribution du Chiffre d'affaires total")
    
    # Affiche l'image
    plt.show()