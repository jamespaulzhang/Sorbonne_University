import math
import random

import matplotlib.pyplot as plt
import pandas as pd
import pulp
import streamlit as st
import trait_donnees


# fréquence 15 et 30 scénarios et seed 42

# Les trois moments où on choisit une équipe.
# 0  = début du Tour
# 10 = après la première vague d'abandons
# 15 = après la deuxième vague d'abandons
PERIODES = [0, 10, 15]

# Les étapes "classiques" du Tour. Le bonus final est rangé à part avec E_FINAL.
ETAPES = list(range(1, 22))
E_FINAL = 22

# Listes de base des coureurs qui abandonnent.
# Elles restent dans le code : l'utilisateur n'a pas besoin de les modifier dans l'interface.
ABANDONS_10 = [
    "Ganna Filippo",
    "Bissegger Stefan",
    "Philipsen Jasper",
    "Jeannière Emilien",
    "De Buyst Jasper",
    "Cattaneo Mattia",
    "Haig Jack",
    "Dunbar Eddie",
    "Almeida Jo",
    "Berg Marijn",
    "Zimmermann Georg",
    "Wærenskjold Søren",
]

ABANDONS_15 = [
    "Bol Cees",
    "Evenepoel Remco",
    "Skjelmose Mattias",
    "Coquard Bryan",
    "Cras Steff",
    "Van Eetvelt Lennert",
    "Poel Mathieu",
]

TAILLE_MIN_SIMULATION = 13
TAILLE_MAX_SIMULATION = 17
TEMPS_LIMITE_CBC = 10
MIP_GAP_RELATIF = 0.0
TEMPS_LIMITE_PAR_SCENARIO = 3
TOLERANCE_OPTIMALITE_SIMULATION = 0.02
TEMPS_LIMITE_SENSIBILITE = 3


def periode(etape):
    # On rattache chaque étape à la période où l'équipe est utilisée.
    if etape <= 10:
        return 0
    if etape <= 15:
        return 10
    return 15


@st.cache_data(show_spinner=False)
def charger_donnees():
    # On ne relit pas les CSV ici.
    # On réutilise directement le travail déjà fait dans trait_donnees.py.
    points, e_final = trait_donnees.data_tour_de_france()
    points = points.copy()

    # c = dictionnaire déjà préparé dans trait_donnees.py : coureur -> prix.
    prix = trait_donnees.c.copy()
    coureurs = list(prix.keys())

    return {
        "base": "trait_donnees.py",
        "coureurs": coureurs,
        "prix": prix,
        "points": points,
    }


def somme_points_periode(points, coureur, periode_cible):
    # Total des points du coureur uniquement sur les étapes de la période demandée.
    return sum(points.get((coureur, e), 0) for e in ETAPES if periode(e) == periode_cible)


def calculer_couches_pareto(df, cout_col="cout_periode", points_col="points_comptes"):
    restants = set(df.index)
    couches = pd.Series(index=df.index, dtype=int)
    couche = 1

    # Idée simple :
    # couche 1 = les coureurs qu'aucun autre coureur ne bat clairement
    # couche 2 = les meilleurs après avoir enlevé la couche 1
    # etc.
    while restants:
        front = []
        for idx in restants:
            row = df.loc[idx]
            domine = False
            for other_idx in restants:
                if other_idx == idx:
                    continue
                other = df.loc[other_idx]
                # "other" domine "row" si :
                # - il coûte moins cher ou pareil
                # - il marque plus de points ou pareil
                # - et il est strictement meilleur sur au moins un des deux critères
                if (
                    other[cout_col] <= row[cout_col]
                    and other[points_col] >= row[points_col]
                    and (
                        other[cout_col] < row[cout_col]
                        or other[points_col] > row[points_col]
                    )
                ):
                    domine = True
                    break
            if not domine:
                front.append(idx)

        for idx in front:
            couches.loc[idx] = couche
        restants -= set(front)
        couche += 1

    return couches.astype(int)


def dataframe_periode_finale(resultat):
    # On prépare une table spéciale pour la dernière période.
    # C'est cette table qui sert aux histogrammes prix / coût / Pareto.
    df = resultat["analyse_df"]
    df = df[df["periode"] == 15].copy()
    df["prix_initial"] = df["coureur"].map(resultat["prix"])
    df["cout_transfert"] = df["coureur"].map(resultat["prix_achat_transfert"])
    df["points_comptes"] = df["points_periode"] + df["bonus_final"]
    df["points_sur_prix"] = df["points_comptes"] / df["prix_initial"]
    df["points_sur_cout"] = df["points_comptes"] / df["cout_transfert"]
    df["selectionne"] = df["selectionne"].astype(int)
    df["couche_pareto"] = calculer_couches_pareto(
        df,
        cout_col="cout_transfert",
        points_col="points_comptes",
    )
    return df


def prefixe_jusqu_au_dernier_selectionne(df, sort_cols, ascending):
    # On trie les coureurs selon un critère.
    # Ensuite on garde tout le monde jusqu'au dernier coureur choisi par le modèle.
    # Comme ça l'histogramme montre aussi les coureurs qui étaient "devant"
    # mais qui n'ont pas été retenus à cause du budget, des abandons, etc.
    ranked = df.sort_values(sort_cols, ascending=ascending).copy()
    ranked["rang"] = range(1, len(ranked) + 1)
    selected_ranks = ranked.loc[ranked["selectionne"] == 1, "rang"]
    if selected_ranks.empty:
        return ranked, len(ranked)
    dernier_rang = int(selected_ranks.max())
    return ranked[ranked["rang"] <= dernier_rang].copy(), dernier_rang


def afficher_histogramme_prefixe(df, value_col, title, ylabel, sort_cols, ascending):
    # Petit code commun pour éviter de réécrire trois fois le même histogramme.
    prefixe, dernier_rang = prefixe_jusqu_au_dernier_selectionne(df, sort_cols, ascending)

    # Rouge = coureur dans l'équipe optimale, bleu = coureur non choisi.
    couleurs = ["#d62728" if selected else "#7fb3d5" for selected in prefixe["selectionne"]]

    # La largeur grandit avec le nombre de coureurs, sinon les noms se marchent dessus.
    fig_width = min(24, max(10, len(prefixe) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    ax.bar(prefixe["coureur"], prefixe[value_col], color=couleurs, edgecolor="black", linewidth=0.6)
    ax.set_title(f"{title} - jusqu'au rang {dernier_rang}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Coureurs")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"{len(prefixe)} coureurs affichés. Rouge = sélectionné dans l'équipe optimale de la période 15."
    )


def resoudre_modele_simulation(coureurs, prix, points, budget_initial, taille_equipe, abandons_10, abandons_15_extra, time_limit, gap_rel, nom="Fantasy_TDF_Simulation"):
    # Cette fonction est la version "légère" du modèle.
    # Elle sert dans la simulation Monte Carlo : on la lance plein de fois,
    # donc elle retourne seulement ce dont on a besoin pour compter les choix.

    # Acheter un coureur pendant un transfert coûte 10% plus cher que son prix de base.
    prix_achat_transfert = {p: math.ceil(1.1 * prix[p]) for p in coureurs}

    # On prépare les abandons en set, parce que tester "p in set" est rapide et propre.
    abandons_10_set = set(abandons_10)
    abandons_15_set = abandons_10_set | set(abandons_15_extra)

    # Problème PuLP : on maximise les points.
    prob = pulp.LpProblem(nom, pulp.LpMaximize)

    # x[p][t] = 1 si le coureur p est dans l'équipe à la période t.
    x = pulp.LpVariable.dicts("x", (coureurs, PERIODES), cat=pulp.LpBinary)

    # achat[p][t] et vente[p][t] servent à représenter les transferts.
    achat = pulp.LpVariable.dicts("achat", (coureurs, PERIODES), cat=pulp.LpBinary)
    vente = pulp.LpVariable.dicts("vente", (coureurs, PERIODES), cat=pulp.LpBinary)

    # budget[t] = argent restant à la période t.
    budget = pulp.LpVariable.dicts("budget", PERIODES, lowBound=0)

    # Objectif :
    # points des étapes + bonus final pour les coureurs présents en période 15.
    prob += (
        pulp.lpSum(
            points.get((p, e), 0) * x[p][periode(e)]
            for p in coureurs
            for e in ETAPES
        )
        + pulp.lpSum(points.get((p, E_FINAL), 0) * x[p][15] for p in coureurs)
    )

    for t in PERIODES:
        # À chaque période, on veut exactement la bonne taille d'équipe.
        prob += pulp.lpSum(x[p][t] for p in coureurs) == taille_equipe

    # Au début, le prix de l'équipe + le budget restant doit faire le budget initial.
    prob += pulp.lpSum(prix[p] * x[p][0] for p in coureurs) + budget[0] == budget_initial

    for i in range(1, len(PERIODES)):
        t_prec = PERIODES[i - 1]
        t = PERIODES[i]

        for p in coureurs:
            # Cette égalité relie la présence dans l'équipe et les transferts :
            # si le coureur arrive, achat = 1 ; s'il part, vente = 1.
            prob += x[p][t] - x[p][t_prec] == achat[p][t] - vente[p][t]

            # Un coureur ne peut pas être acheté et vendu en même temps.
            prob += achat[p][t] + vente[p][t] <= 1

            # Ces contraintes évitent les achats/ventes absurdes.
            prob += achat[p][t] <= 1 - x[p][t_prec]
            prob += achat[p][t] <= x[p][t]
            prob += vente[p][t] <= x[p][t_prec]
            prob += vente[p][t] <= 1 - x[p][t]

        # Mise à jour du budget :
        # on paie les entrants au prix transfert, et on récupère le prix des sortants.
        prob += (
            budget[t]
            == budget[t_prec]
            - pulp.lpSum(prix_achat_transfert[p] * achat[p][t] for p in coureurs)
            + pulp.lpSum(prix[p] * vente[p][t] for p in coureurs)
        )

    for p in coureurs:
        # On n'a le droit de vendre que les coureurs qui ont abandonné.
        if p not in abandons_10_set:
            prob += vente[p][10] == 0
        if p not in abandons_15_set:
            prob += vente[p][15] == 0

        # Si un coureur a abandonné, il ne peut plus être dans l'équipe ensuite.
        if p in abandons_10_set:
            prob += x[p][10] == 0
            prob += x[p][15] == 0
        elif p in abandons_15_set:
            prob += x[p][15] == 0

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=gap_rel)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    def val(variable):
        # PuLP peut renvoyer None si la variable n'a pas de valeur.
        # On sécurise pour éviter de casser l'affichage ou la simulation.
        raw = pulp.value(variable)
        return 0.0 if raw is None else float(raw)

    # On extrait les équipes trouvées par CBC.
    equipes = {
        t: [p for p in coureurs if val(x[p][t]) > 0.5]
        for t in PERIODES
    }

    return {
        "status": status,
        "objectif": val(prob.objective),
        "equipes": equipes,
        "taille_equipe": taille_equipe,
    }


def simuler_facteur_chance(budget_initial, taille_equipe_base, abandons_10, abandons_15_extra, n_scenarios, sigma_points, sigma_couts, sigma_taille, taille_min, taille_max, time_limit, gap_rel, seed, progress_callback=None):
    # La simulation "chance" relance le modèle plusieurs fois.
    # À chaque scénario, on bouge un peu les points, les coûts et la taille d'équipe.
    # À la fin, on compte quels coureurs reviennent le plus souvent.

    # Le seed permet de refaire exactement la même simulation si besoin.
    rng = random.Random(seed)

    # On repart toujours des données officielles préparées dans trait_donnees.py.
    data = charger_donnees()
    coureurs = data["coureurs"]
    prix_base = data["prix"]
    points_base = data["points"]

    # Compteurs :
    # - par période : combien de fois le coureur est choisi en P0, P10, P15
    # - total : nombre de sélections toutes périodes confondues
    # - scénario : nombre de scénarios où il apparaît au moins une fois
    selection_par_periode = {
        t: {p: 0 for p in coureurs}
        for t in PERIODES
    }
    selection_total = {p: 0 for p in coureurs}
    selection_scenario = {p: 0 for p in coureurs}
    scenario_rows = []

    for scenario in range(1, n_scenarios + 1):
        # Taille d'équipe tirée avec une loi normale, puis bornée entre min et max.
        taille_equipe = int(round(rng.gauss(taille_equipe_base, sigma_taille)))
        taille_equipe = max(taille_min, min(taille_max, taille_equipe))

        # Prix simulés :
        # on multiplie chaque prix par un facteur aléatoire autour de 1.
        prix_sim = {}
        for p in coureurs:
            facteur = max(0.1, rng.gauss(1.0, sigma_couts))
            prix_sim[p] = max(1, int(round(prix_base[p] * facteur)))

        # Points simulés :
        # même principe, mais on interdit les points négatifs.
        points_sim = {}
        for key, value in points_base.items():
            facteur = max(0.0, rng.gauss(1.0, sigma_points))
            points_sim[key] = value * facteur

        # On résout le problème avec ces nouvelles données simulées.
        solution = resoudre_modele_simulation(
            coureurs=coureurs,
            prix=prix_sim,
            points=points_sim,
            budget_initial=budget_initial,
            taille_equipe=taille_equipe,
            abandons_10=abandons_10,
            abandons_15_extra=abandons_15_extra,
            time_limit=time_limit,
            gap_rel=gap_rel,
            nom=f"Fantasy_TDF_Simulation_{scenario}",
        )

        scenario_rows.append(
            {
                "scenario": scenario,
                "status": solution["status"],
                "taille_equipe": taille_equipe,
                "objectif": solution["objectif"],
            }
        )

        if solution["status"] == "Optimal":
            # Si CBC a trouvé l'optimum, on compte les coureurs sélectionnés.
            deja_selectionne = set()
            for t in PERIODES:
                for p in solution["equipes"][t]:
                    selection_par_periode[t][p] += 1
                    selection_total[p] += 1
                    deja_selectionne.add(p)
            for p in deja_selectionne:
                selection_scenario[p] += 1

        if progress_callback:
            # Petit retour visuel dans Streamlit pendant que ça tourne.
            progress_callback(scenario, n_scenarios, solution["status"])

    scenarios_df = pd.DataFrame(scenario_rows)

    # On calcule les fréquences seulement sur les scénarios vraiment optimaux.
    nb_optimaux = int((scenarios_df["status"] == "Optimal").sum()) if not scenarios_df.empty else 0
    denom = max(nb_optimaux, 1)

    rows = []
    for p in coureurs:
        # Total de points "de base", juste pour aider à lire le classement final.
        points_total_base = sum(points_base.get((p, e), 0) for e in ETAPES) + points_base.get((p, E_FINAL), 0)
        rows.append(
            {
                "coureur": p,
                "prix_base": prix_base[p],
                "points_base": points_total_base,
                "choisi_total": selection_total[p],
                "choisi_au_moins_une_periode": selection_scenario[p],
                "choisi_p0": selection_par_periode[0][p],
                "choisi_p10": selection_par_periode[10][p],
                "choisi_p15": selection_par_periode[15][p],
                "freq_total_par_slot": selection_total[p] / (denom * len(PERIODES)),
                "freq_scenario": selection_scenario[p] / denom,
                "freq_p0": selection_par_periode[0][p] / denom,
                "freq_p10": selection_par_periode[10][p] / denom,
                "freq_p15": selection_par_periode[15][p] / denom,
            }
        )

    # Classement final : les coureurs qui reviennent le plus souvent sont en haut.
    comptage_df = pd.DataFrame(rows).sort_values(
        ["choisi_total", "choisi_p15", "points_base"],
        ascending=[False, False, False],
    )

    return {
        "comptage_df": comptage_df,
        "scenarios_df": scenarios_df,
        "nb_scenarios": n_scenarios,
        "nb_optimaux": nb_optimaux,
        "sigma_points": sigma_points,
        "sigma_couts": sigma_couts,
        "sigma_taille": sigma_taille,
        "seed": seed,
    }


def optimiser_equipe_pulp(budget_initial, taille_equipe, abandons_10, abandons_15_extra, time_limit, gap_rel, force_coureur=None, force_periode=None, interdire_coureur=None, interdire_periode=None, autoriser_transferts=True):
    # Fonction principale de l'app :
    # elle construit le modèle PuLP complet, le résout, puis prépare toutes
    # les tables dont Streamlit a besoin pour afficher les résultats.

    # Données du projet, déjà préparées dans trait_donnees.py.
    data = charger_donnees()
    coureurs = data["coureurs"]
    prix = data["prix"]
    points = data["points"]

    # En transfert, acheter coûte 10% plus cher.
    prix_achat_transfert = {p: math.ceil(1.1 * prix[p]) for p in coureurs}

    # Les abandons sont convertis en ensembles pour simplifier les tests.
    abandons_10_set = set(abandons_10)
    abandons_15_set = abandons_10_set | set(abandons_15_extra)

    # Si un nom tapé dans l'interface n'existe pas dans les données, on le signale.
    coureurs_set = set(coureurs)
    inconnus = sorted((abandons_10_set | abandons_15_set) - coureurs_set)

    # Création du problème : on maximise les points fantasy.
    prob = pulp.LpProblem("Fantasy_TDF_PuLP", pulp.LpMaximize)

    # x[p][t] = 1 si le coureur p est choisi à la période t.
    x = pulp.LpVariable.dicts("x", (coureurs, PERIODES), cat=pulp.LpBinary)

    # achat/vente servent à suivre les transferts entre deux périodes.
    achat = pulp.LpVariable.dicts("achat", (coureurs, PERIODES), cat=pulp.LpBinary)
    vente = pulp.LpVariable.dicts("vente", (coureurs, PERIODES), cat=pulp.LpBinary)

    # budget[t] garde l'argent disponible après les achats/ventes.
    budget = pulp.LpVariable.dicts("budget", PERIODES, lowBound=0)

    # Objectif du modèle :
    # - les points de chaque étape comptent pour l'équipe active à ce moment-là
    # - le bonus final compte seulement pour l'équipe de la dernière période
    prob += (
        pulp.lpSum(
            points.get((p, e), 0) * x[p][periode(e)]
            for p in coureurs
            for e in ETAPES
        )
        + pulp.lpSum(points.get((p, E_FINAL), 0) * x[p][15] for p in coureurs)
    )

    for t in PERIODES:
        # On force exactement "taille_equipe" coureurs dans l'équipe à chaque période.
        prob += pulp.lpSum(x[p][t] for p in coureurs) == taille_equipe

    # Budget initial : prix des coureurs choisis au départ + argent restant.
    prob += (
        pulp.lpSum(prix[p] * x[p][0] for p in coureurs) + budget[0] == budget_initial,
        "Budget_initial",
    )

    for i in range(1, len(PERIODES)):
        t_prec = PERIODES[i - 1]
        t = PERIODES[i]

        for p in coureurs:
            # Si x change entre deux périodes, ça doit passer par achat ou vente.
            prob += x[p][t] - x[p][t_prec] == achat[p][t] - vente[p][t]

            # Pas d'achat et vente en même temps pour le même coureur.
            prob += achat[p][t] + vente[p][t] <= 1

            # Mode sans transfert :
            # on bloque tous les achats et toutes les ventes.
            # Du coup, l'équipe doit rester la même sur les périodes.
            if not autoriser_transferts:
                prob += achat[p][t] == 0
                prob += vente[p][t] == 0

            # On ne peut acheter que quelqu'un qui n'était pas là avant,
            # et vendre que quelqu'un qui était déjà dans l'équipe.
            prob += achat[p][t] <= 1 - x[p][t_prec]
            prob += achat[p][t] <= x[p][t]
            prob += vente[p][t] <= x[p][t_prec]
            prob += vente[p][t] <= 1 - x[p][t]

        # Budget après transferts.
        prob += (
            budget[t]
            == budget[t_prec]
            - pulp.lpSum(prix_achat_transfert[p] * achat[p][t] for p in coureurs)
            + pulp.lpSum(prix[p] * vente[p][t] for p in coureurs)
        )

    for p in coureurs:
        # Règle du projet : on vend uniquement les coureurs qui abandonnent.
        if p not in abandons_10_set:
            prob += vente[p][10] == 0
        if p not in abandons_15_set:
            prob += vente[p][15] == 0

        # Un coureur abandonné ne peut plus être aligné après son abandon.
        if p in abandons_10_set:
            prob += x[p][10] == 0
            prob += x[p][15] == 0
        elif p in abandons_15_set:
            prob += x[p][15] == 0

    # Petit outil d'analyse :
    # si on veut comprendre pourquoi un coureur n'est pas choisi,
    # on peut le forcer dans une période et regarder combien de points on perd.
    if force_coureur in coureurs and force_periode in PERIODES:
        prob += x[force_coureur][force_periode] == 1

    # Même idée, mais dans l'autre sens :
    # on interdit un coureur pour mesurer ce que l'équipe perd sans lui.
    if interdire_coureur in coureurs and interdire_periode in PERIODES:
        prob += x[interdire_coureur][interdire_periode] == 0

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=gap_rel)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]

    def val(variable):
        # Petite sécurité : si PuLP n'a pas de valeur, on met 0.
        raw = pulp.value(variable)
        return 0.0 if raw is None else float(raw)

    def is_selected(variable):
        # Les variables binaires ressortent souvent comme 0.0 ou 1.0.
        # On utilise 0.5 comme seuil classique.
        return val(variable) > 0.5

    # Extraction lisible des équipes, budgets et transferts.
    equipes = {
        t: [p for p in coureurs if is_selected(x[p][t])]
        for t in PERIODES
    }
    budgets = {t: val(budget[t]) for t in PERIODES}
    transferts = {
        t: {
            "entrants": [p for p in coureurs if is_selected(achat[p][t])],
            "sortants": [p for p in coureurs if is_selected(vente[p][t])],
        }
        for t in PERIODES[1:]
    }

    points_rows = []
    for p in coureurs:
        # Table complète des points par coureur, pour l'onglet "Points et scores".
        row = {
            "Coureur": p,
            "Prix": prix[p],
            "Prix achat transfert": prix_achat_transfert[p],
            "Période 0": somme_points_periode(points, p, 0),
            "Période 10": somme_points_periode(points, p, 10),
            "Période 15": somme_points_periode(points, p, 15),
            "Bonus final": points.get((p, E_FINAL), 0),
        }
        for e in ETAPES:
            row[f"Étape {e}"] = points.get((p, e), 0)
        row["Total"] = row["Période 0"] + row["Période 10"] + row["Période 15"] + row["Bonus final"]
        points_rows.append(row)
    points_df = pd.DataFrame(points_rows).set_index("Coureur")

    equipes_rows = []
    for t in PERIODES:
        for p in equipes[t]:
            # Table des coureurs réellement choisis par le modèle.
            bonus_final = points.get((p, E_FINAL), 0) if t == 15 else 0
            points_periode = somme_points_periode(points, p, t)
            equipes_rows.append(
                {
                    "Période": t,
                    "Coureur": p,
                    "Prix initial": prix[p],
                    "Prix achat transfert": prix_achat_transfert[p],
                    "Points période": points_periode,
                    "Bonus final": bonus_final,
                    "Points comptés": points_periode + bonus_final,
                }
            )
    equipes_df = pd.DataFrame(equipes_rows)

    transferts_rows = []
    for t, tr in transferts.items():
        for p in tr["entrants"]:
            # Un entrant coûte le prix transfert.
            transferts_rows.append(
                {
                    "Période": t,
                    "Type": "entrant",
                    "Coureur": p,
                    "Montant": prix_achat_transfert[p],
                }
            )
        for p in tr["sortants"]:
            # Un sortant rapporte son prix initial.
            transferts_rows.append(
                {
                    "Période": t,
                    "Type": "sortant",
                    "Coureur": p,
                    "Montant": prix[p],
                }
            )
    transferts_df = pd.DataFrame(transferts_rows)

    scores_rows = []
    for e in ETAPES:
        # Score étape par étape de l'équipe active.
        t = periode(e)
        score = sum(points.get((p, e), 0) for p in equipes[t])
        scores_rows.append({"Étape": e, "Période": t, "Score": score})
    scores_df = pd.DataFrame(scores_rows)

    # Contrôle simple : on recalcule le score depuis les tables,
    # histoire de comparer avec l'objectif donné par PuLP.
    bonus_final = sum(points.get((p, E_FINAL), 0) for p in equipes[15])
    score_recalcule = float(scores_df["Score"].sum() + bonus_final)
    objectif = val(prob.objective)

    analyse_rows = []
    for t in PERIODES:
        for p in coureurs:
            # Grande table "analyse" : une ligne par coureur et par période.
            # Elle sert surtout aux histogrammes et à l'export CSV.
            analyse_rows.append(
                {
                    "coureur": p,
                    "periode": t,
                    "selectionne": int(p in equipes[t]),
                    "cout_periode": prix[p] if t == 0 else prix_achat_transfert[p],
                    "points_periode": somme_points_periode(points, p, t),
                    "points_restants": sum(
                        points.get((p, e), 0)
                        for e in ETAPES
                        if periode(e) >= t
                    ),
                    "bonus_final": points.get((p, E_FINAL), 0) if t == 15 else 0,
                    "transferrable_10": int(p in abandons_10_set),
                    "transferrable_15": int(p in abandons_15_set),
                }
            )
    analyse_df = pd.DataFrame(analyse_rows)

    return {
        "status": status,
        "objectif": objectif,
        "score_recalcule": score_recalcule,
        "bonus_final": bonus_final,
        "budget_initial": budget_initial,
        "taille_equipe": taille_equipe,
        "equipes": equipes,
        "budgets": budgets,
        "transferts": transferts,
        "points_df": points_df,
        "equipes_df": equipes_df,
        "transferts_df": transferts_df,
        "scores_df": scores_df,
        "analyse_df": analyse_df,
        "prix": prix,
        "prix_achat_transfert": prix_achat_transfert,
        "inconnus": inconnus,
        "data_dir": data["base"],
        "autoriser_transferts": autoriser_transferts,
    }


def ligne_analyse_p15(resultat, coureur, etiquette):
    # On récupère les chiffres utiles pour expliquer la période 15.
    lignes_coureur = resultat["analyse_df"][resultat["analyse_df"]["coureur"] == coureur].set_index("periode")
    points_periode_0 = float(lignes_coureur.loc[0, "points_periode"])
    points_periode_10 = float(lignes_coureur.loc[10, "points_periode"])
    points_periode_15 = float(lignes_coureur.loc[15, "points_periode"])
    bonus_final = float(lignes_coureur.loc[15, "bonus_final"])

    return {
        "Coureur": coureur,
        "Situation": etiquette,
        "Prix initial": resultat["prix"][coureur],
        "Coût transfert": resultat["prix_achat_transfert"][coureur],
        "Points période 0": points_periode_0,
        "Points période 10": points_periode_10,
        "Points période 15": points_periode_15,
        "Bonus final": bonus_final,
        "Points comptés P15": points_periode_15 + bonus_final,
        "Dans base P15": coureur in resultat["equipes"][15],
    }


def afficher_analyse_coureur_force(resultat_base, coureur, budget_initial, taille_equipe):
    if resultat_base is None:
        st.info("Lancez d'abord l'optimisation de base pour avoir une comparaison.")
        return

    if coureur not in resultat_base["prix"]:
        st.warning("Ce coureur n'existe pas dans les données chargées.")
        return

    with st.spinner(f"Analyse de {coureur} en période 15..."):
        resultat_force = optimiser_equipe_pulp(
            budget_initial=budget_initial,
            taille_equipe=taille_equipe,
            abandons_10=ABANDONS_10,
            abandons_15_extra=ABANDONS_15,
            time_limit=TEMPS_LIMITE_CBC,
            gap_rel=MIP_GAP_RELATIF,
            force_coureur=coureur,
            force_periode=15,
            autoriser_transferts=resultat_base.get("autoriser_transferts", True),
        )

    st.markdown(f"### Analyse de {coureur}")

    if resultat_force["status"] != "Optimal":
        st.warning(f"Impossible de trouver une solution optimale en forçant {coureur}. Statut CBC : {resultat_force['status']}.")
        return

    score_base = resultat_base["objectif"]
    score_force = resultat_force["objectif"]
    ecart = score_force - score_base
    deja_dans_base = coureur in resultat_base["equipes"][15]

    col1, col2, col3 = st.columns(3)
    col1.metric("Score de base", f"{score_base:.0f}")
    col2.metric("Score avec ce coureur", f"{score_force:.0f}")
    col3.metric("Écart", f"{ecart:.0f}")

    if deja_dans_base:
        st.success(f"{coureur} est déjà dans l'équipe optimale de base en période 15.")
    elif ecart == 0:
        st.success(f"{coureur} peut rentrer sans perdre de points : il existe une autre solution optimale.")
    else:
        st.warning(f"En forçant {coureur}, le modèle perd {abs(ecart):.0f} points. Il est proche, mais pas assez fort dans les données de base.")

    base_p15 = set(resultat_base["equipes"][15])
    force_p15 = set(resultat_force["equipes"][15])
    sortants = sorted(base_p15 - force_p15)
    entrants = sorted(force_p15 - base_p15)

    st.write("Changement en période 15 :")
    if entrants:
        st.write("Entre :", ", ".join(entrants))
    if sortants:
        st.write("Sort :", ", ".join(sortants))
    if not entrants and not sortants:
        st.write("Aucun changement : le coureur était déjà dans l'équipe.")

    lignes = []
    coureurs_a_comparer = []
    for p in [coureur] + sortants + entrants:
        if p not in coureurs_a_comparer:
            coureurs_a_comparer.append(p)

    for p in coureurs_a_comparer:
        if p == coureur:
            etiquette = "Coureur analysé"
        elif p in sortants:
            etiquette = "Sort de l'équipe"
        elif p in entrants:
            etiquette = "Entre avec la solution forcée"
        else:
            etiquette = "Comparaison"
        ligne = ligne_analyse_p15(resultat_base, p, etiquette)
        ligne["Dans solution forcée P15"] = p in force_p15
        lignes.append(ligne)

    comparaison_df = pd.DataFrame(lignes)
    st.dataframe(comparaison_df, use_container_width=True, hide_index=True)

    st.info(
        "Lecture simple : si le score baisse peu, le coureur est un candidat presque optimal. "
        "Il peut donc revenir souvent dans la simulation dès que les points, les coûts ou la taille d'équipe bougent un peu."
    )


def points_comptes_coureur_periode(resultat, coureur, periode_cible):
    ligne = resultat["analyse_df"][
        (resultat["analyse_df"]["coureur"] == coureur)
        & (resultat["analyse_df"]["periode"] == periode_cible)
    ].iloc[0]
    return float(ligne["points_periode"] + ligne["bonus_final"])


def candidats_sensibilite_periode(resultat, periode_cible):
    # On classe les coureurs par points de la période.
    # Pour les non-choisis, on s'arrête au dernier coureur choisi dans ce classement.
    df = resultat["analyse_df"][resultat["analyse_df"]["periode"] == periode_cible].copy()
    df["points_comptes"] = df["points_periode"] + df["bonus_final"]
    df = df.sort_values(
        ["points_comptes", "cout_periode", "coureur"],
        ascending=[False, True, True],
    ).copy()
    df["rang"] = range(1, len(df) + 1)

    rangs_selectionnes = df.loc[df["selectionne"] == 1, "rang"]
    dernier_rang_selectionne = int(rangs_selectionnes.max()) if not rangs_selectionnes.empty else len(df)
    prefixe = df[df["rang"] <= dernier_rang_selectionne].copy()

    selectionnes = df[df["selectionne"] == 1].copy()
    non_selectionnes = prefixe[prefixe["selectionne"] == 0].copy()
    return selectionnes, non_selectionnes, dernier_rang_selectionne


def ligne_sensibilite_sortie(resultat_base, coureur, periode_cible):
    # Candidat déjà choisi : on l'interdit, puis on regarde combien de points l'équipe perd.
    # Cela donne l'alpha à partir duquel il risque de sortir si ses points baissent.
    points_periode = points_comptes_coureur_periode(resultat_base, coureur, periode_cible)
    resultat_sans = optimiser_equipe_pulp(
        budget_initial=resultat_base["budget_initial"],
        taille_equipe=resultat_base["taille_equipe"],
        abandons_10=ABANDONS_10,
        abandons_15_extra=ABANDONS_15,
        time_limit=TEMPS_LIMITE_SENSIBILITE,
        gap_rel=MIP_GAP_RELATIF,
        interdire_coureur=coureur,
        interdire_periode=periode_cible,
        autoriser_transferts=resultat_base.get("autoriser_transferts", True),
    )

    if resultat_sans["status"] != "Optimal" or points_periode <= 0:
        return {
            "Période": periode_cible,
            "Coureur": coureur,
            "Type": "Déjà choisi",
            "Points période": points_periode,
            "Score base": resultat_base["objectif"],
            "Score alternatif": None,
            "Écart score": None,
            "Alpha critique": None,
            "Variation": None,
            "Interprétation": "Non calculable",
        }

    ecart_score = resultat_base["objectif"] - resultat_sans["objectif"]
    alpha_brut = 1 - (ecart_score / points_periode)
    alpha_affiche = max(0, min(1, alpha_brut))
    variation = (alpha_affiche - 1) * 100

    if alpha_brut <= 0:
        interpretation = "Très solide"
    elif alpha_brut >= 1:
        interpretation = "Très fragile"
    else:
        interpretation = f"Sort sous {100 * alpha_affiche:.1f}%"

    return {
        "Période": periode_cible,
        "Coureur": coureur,
        "Type": "Déjà choisi",
        "Points période": points_periode,
        "Score base": resultat_base["objectif"],
        "Score alternatif": resultat_sans["objectif"],
        "Écart score": ecart_score,
        "Alpha critique": alpha_affiche,
        "Variation": variation,
        "Interprétation": interpretation,
    }


def ligne_sensibilite_entree(resultat_base, coureur, periode_cible, rang):
    # Candidat non choisi : on le force, puis on regarde combien il manque.
    # Cela donne l'alpha nécessaire pour qu'il puisse entrer.
    points_periode = points_comptes_coureur_periode(resultat_base, coureur, periode_cible)
    resultat_force = optimiser_equipe_pulp(
        budget_initial=resultat_base["budget_initial"],
        taille_equipe=resultat_base["taille_equipe"],
        abandons_10=ABANDONS_10,
        abandons_15_extra=ABANDONS_15,
        time_limit=TEMPS_LIMITE_SENSIBILITE,
        gap_rel=MIP_GAP_RELATIF,
        force_coureur=coureur,
        force_periode=periode_cible,
        autoriser_transferts=resultat_base.get("autoriser_transferts", True),
    )

    if resultat_force["status"] != "Optimal" or points_periode <= 0:
        return {
            "Période": periode_cible,
            "Rang": rang,
            "Coureur": coureur,
            "Type": "Non choisi",
            "Points période": points_periode,
            "Score base": resultat_base["objectif"],
            "Score alternatif": None,
            "Écart score": None,
            "Alpha critique": None,
            "Variation": None,
            "Interprétation": "Non calculable",
        }

    ecart_score = resultat_base["objectif"] - resultat_force["objectif"]
    if ecart_score <= 0:
        alpha = 1.0
        interpretation = "Peut déjà entrer"
    else:
        alpha = 1 + (ecart_score / points_periode)
        interpretation = f"Entre à {100 * alpha:.1f}%"

    return {
        "Période": periode_cible,
        "Rang": rang,
        "Coureur": coureur,
        "Type": "Non choisi",
        "Points période": points_periode,
        "Score base": resultat_base["objectif"],
        "Score alternatif": resultat_force["objectif"],
        "Écart score": ecart_score,
        "Alpha critique": alpha,
        "Variation": (alpha - 1) * 100,
        "Interprétation": interpretation,
    }


def calculer_sensibilite_points(resultat_base, progress_callback=None):
    lignes_sortie = []
    lignes_entree = []
    candidats_par_periode = {}
    total = 0

    for periode_cible in PERIODES:
        selectionnes, non_selectionnes, dernier_rang = candidats_sensibilite_periode(resultat_base, periode_cible)
        candidats_par_periode[periode_cible] = (selectionnes, non_selectionnes, dernier_rang)
        total += len(selectionnes) + len(non_selectionnes)

    fait = 0
    for periode_cible in PERIODES:
        selectionnes, non_selectionnes, dernier_rang = candidats_par_periode[periode_cible]

        for _, row in selectionnes.iterrows():
            fait += 1
            if progress_callback:
                progress_callback(fait, total, f"Période {periode_cible} - baisse de {row['coureur']}")
            lignes_sortie.append(ligne_sensibilite_sortie(resultat_base, row["coureur"], periode_cible))

        for _, row in non_selectionnes.iterrows():
            fait += 1
            if progress_callback:
                progress_callback(fait, total, f"Période {periode_cible} - hausse de {row['coureur']}")
            lignes_entree.append(ligne_sensibilite_entree(resultat_base, row["coureur"], periode_cible, int(row["rang"])))

    return {
        "sortie_df": pd.DataFrame(lignes_sortie),
        "entree_df": pd.DataFrame(lignes_entree),
        "total_tests": total,
    }


def nom_periode_analysee(periode_cible):
    if periode_cible == 0:
        return "Période 0 : étapes 1 à 10"
    if periode_cible == 10:
        return "Période 10 : étapes 11 à 15"
    return "Période 15 : étapes 16 à 21 + bonus final"


def formater_pourcentage(valeur):
    if valeur is None or pd.isna(valeur):
        return ""
    return f"{100 * float(valeur):.1f}%"


def formater_variation(valeur):
    if valeur is None or pd.isna(valeur):
        return ""
    return f"{float(valeur):+.1f}%"


def formater_nombre(valeur):
    if valeur is None or pd.isna(valeur):
        return ""
    return f"{float(valeur):.0f}"


def tableau_sensibilite_affichage(df, periode_cible, avec_rang=False):
    # Le tableau brut contient beaucoup de colonnes techniques.
    # Ici on prépare une version plus lisible pour l'interface.
    if df.empty:
        return df

    affichage = df.copy()
    type_ligne = affichage["Type"].iloc[0] if "Type" in affichage.columns else ""
    colonne_ecart = "Points à gagner" if type_ligne == "Non choisi" else "La perte de score si interdiction"
    affichage["Période analysée"] = nom_periode_analysee(periode_cible)
    affichage["Points de la période"] = affichage["Points période"].apply(formater_nombre)
    affichage[colonne_ecart] = affichage["Écart score"].apply(formater_nombre)
    affichage["Alpha critique"] = affichage["Alpha critique"].apply(formater_pourcentage)
    affichage["Variation nécessaire"] = affichage["Variation"].apply(formater_variation)

    colonnes = [
        "Période analysée",
        "Coureur",
        "Points de la période",
        colonne_ecart,
        "Alpha critique",
        "Variation nécessaire",
        "Interprétation",
    ]
    if avec_rang:
        colonnes.insert(1, "Rang")

    return affichage[colonnes]


def afficher_sensibilite_points(resultat):
    st.markdown("### Analyse de sensibilité des points")
    st.write(
        "On cherche un facteur alpha par coureur et par période. "
        "Pour un coureur déjà choisi, alpha indique à partir de quelle baisse il sort. "
        "Pour un coureur non choisi, alpha indique à partir de quelle hausse il entre."
    )
    st.caption(
        "Exemple de lecture : alpha = 97% veut dire que le coureur sort si ses points de la période "
        "descendent sous 97% de leur valeur actuelle. Alpha = 103% veut dire qu'un coureur non choisi "
        "entre si ses points montent à 103%."
    )

    if st.button("Lancer l'analyse de sensibilité des points"):
        progress = st.progress(0)
        status_box = st.empty()

        def on_progress(done, total, message):
            progress.progress(done / max(total, 1))
            status_box.caption(f"{done}/{total} - {message}")

        with st.spinner("Analyse de sensibilité en cours..."):
            sensibilite = calculer_sensibilite_points(resultat, progress_callback=on_progress)

        progress.empty()
        status_box.empty()
        st.session_state["sensibilite_points"] = sensibilite

    if "sensibilite_points" not in st.session_state:
        st.info("Clique sur le bouton pour calculer les seuils alpha.")
        return

    sensibilite = st.session_state["sensibilite_points"]
    sortie_df = sensibilite["sortie_df"]
    entree_df = sensibilite["entree_df"]

    st.caption(f"Nombre de tests réalisés : {sensibilite['total_tests']}")
    tabs = st.tabs(["Période 0", "Période 10", "Période 15"])

    for tab, periode_cible in zip(tabs, PERIODES):
        with tab:
            st.markdown(f"### {nom_periode_analysee(periode_cible)}")
            st.caption("Les points modifiés dans cette partie concernent uniquement cette période.")

            st.markdown("#### Coureurs déjà dans l'équipe")
            df_sortie = sortie_df[sortie_df["Période"] == periode_cible].sort_values("Alpha critique", ascending=False)
            st.dataframe(
                tableau_sensibilite_affichage(df_sortie, periode_cible),
                use_container_width=True,
                hide_index=True,
                height=360,
            )

            st.markdown("#### Coureurs non choisis jusqu'au dernier choisi")
            df_entree = entree_df[entree_df["Période"] == periode_cible].sort_values("Alpha critique", ascending=True)
            st.dataframe(
                tableau_sensibilite_affichage(df_entree, periode_cible, avec_rang=True),
                use_container_width=True,
                hide_index=True,
                height=360,
            )


def afficher_resultats(resultat):
    # Affichage de l'optimisation classique dans Streamlit.
    mode = "avec transferts" if resultat.get("autoriser_transferts", True) else "sans transfert"
    st.subheader(f"Résultat - analyse {mode}")

    # Les 4 chiffres importants, visibles directement en haut.
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Statut CBC", resultat["status"])
    col2.metric("Objectif", f"{resultat['objectif']:.0f}")
    col3.metric("Score recalculé", f"{resultat['score_recalcule']:.0f}")
    col4.metric("Bonus final", f"{resultat['bonus_final']:.0f}")

    if resultat["inconnus"]:
        # Pratique quand un nom d'abandon est mal écrit.
        st.warning(
            "Noms d'abandons non trouvés dans valeurs.csv : "
            + ", ".join(resultat["inconnus"])
        )

    # Les onglets évitent d'avoir une page interminable.
    tab_equipes, tab_transferts, tab_points, tab_histos, tab_sensibilite, tab_export = st.tabs(
        ["Équipes", "Transferts", "Points et scores", "Histogrammes P15", "Sensibilité points", "Export analyse"]
    )

    with tab_equipes:
        # Une table par période : équipe P0, équipe P10, équipe P15.
        for t in PERIODES:
            st.markdown(f"### Période {t}")
            team_df = resultat["equipes_df"]
            team_df = team_df[team_df["Période"] == t].copy()
            st.dataframe(team_df, use_container_width=True, hide_index=True)
            st.write(f"Budget restant : {resultat['budgets'][t]:.2f}")

    with tab_transferts:
        # Les transferts montrent qui rentre, qui sort, et combien ça coûte/rapporte.
        if resultat["transferts_df"].empty:
            st.info("Aucun transfert.")
        else:
            st.dataframe(resultat["transferts_df"], use_container_width=True, hide_index=True)
            for t in PERIODES[1:]:
                tr = resultat["transferts"][t]
                cout_achats = sum(resultat["prix_achat_transfert"][p] for p in tr["entrants"])
                revenu_ventes = sum(resultat["prix"][p] for p in tr["sortants"])
                st.write(
                    f"Période {t} : achats = {cout_achats:.0f}, "
                    f"ventes = {revenu_ventes:.0f}, net = {cout_achats - revenu_ventes:.0f}"
                )

    with tab_points:
        # Détail des points : utile pour vérifier pourquoi un coureur est intéressant.
        st.markdown("### Points par coureur")
        st.dataframe(resultat["points_df"], use_container_width=True)
        st.markdown("### Score par étape")
        st.dataframe(resultat["scores_df"], use_container_width=True, hide_index=True)
        chart_df = resultat["scores_df"].set_index("Étape")["Score"]
        st.bar_chart(chart_df)

    with tab_histos:
        # Les trois graphiques demandés pour la dernière période.
        st.markdown("### Dernière période : critères séparés")
        df_p15 = dataframe_periode_finale(resultat)

        afficher_histogramme_prefixe(
            df=df_p15,
            value_col="points_sur_prix",
            title="Points / prix initial",
            ylabel="Points / prix",
            sort_cols=["points_sur_prix", "points_comptes", "coureur"],
            ascending=[False, False, True],
        )

        afficher_histogramme_prefixe(
            df=df_p15,
            value_col="points_sur_cout",
            title="Points / coût d'achat en période 15",
            ylabel="Points / coût avec +10%",
            sort_cols=["points_sur_cout", "points_comptes", "coureur"],
            ascending=[False, False, True],
        )

        afficher_histogramme_prefixe(
            df=df_p15,
            value_col="couche_pareto",
            title="Couches de Pareto",
            ylabel="Couche Pareto (1 = meilleur front)",
            sort_cols=["couche_pareto", "points_comptes", "cout_transfert", "coureur"],
            ascending=[True, False, True, True],
        )

        colonnes = [
            "rang",
            "coureur",
            "selectionne",
            "prix_initial",
            "cout_transfert",
            "points_comptes",
            "points_sur_prix",
            "points_sur_cout",
            "couche_pareto",
        ]

        # Tableau associé au graphe Pareto, pour voir les valeurs exactes.
        table_pareto, _ = prefixe_jusqu_au_dernier_selectionne(
            df_p15,
            ["couche_pareto", "points_comptes", "cout_transfert", "coureur"],
            [True, False, True, True],
        )
        table_pareto_affichage = table_pareto[colonnes].rename(
            columns={
                "rang": "Rang",
                "coureur": "Coureur",
                "selectionne": "Sélectionné",
                "prix_initial": "Prix initial",
                "cout_transfert": "Coût transfert",
                "points_comptes": "Points comptés",
                "points_sur_prix": "Points / prix",
                "points_sur_cout": "Points / coût",
                "couche_pareto": "Couche Pareto",
            }
        )
        st.markdown("### Tableau Pareto jusqu'au dernier sélectionné")
        st.dataframe(table_pareto_affichage, use_container_width=True, hide_index=True)

    with tab_sensibilite:
        afficher_sensibilite_points(resultat)

    with tab_export:
        # Export CSV pour réutiliser les résultats ailleurs si besoin.
        st.markdown("### Données pour Pareto / analyse post-optimale")
        analyse_affichage = resultat["analyse_df"].rename(
            columns={
                "coureur": "Coureur",
                "periode": "Période",
                "selectionne": "Sélectionné",
                "cout_periode": "Coût période",
                "points_periode": "Points période",
                "points_restants": "Points restants",
                "bonus_final": "Bonus final",
                "transferrable_10": "Transférable 10",
                "transferrable_15": "Transférable 15",
            }
        )
        st.dataframe(analyse_affichage, use_container_width=True, hide_index=True)
        csv_bytes = resultat["analyse_df"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "Télécharger valeurs_coureurs_par_periode.csv",
            data=csv_bytes,
            file_name="valeurs_coureurs_par_periode.csv",
            mime="text/csv",
        )


def afficher_simulation_chance(resultat, resultat_base=None, budget_initial=None, taille_equipe=None):
    # Affichage des résultats de la simulation Monte Carlo.
    st.subheader("Simulation - facteur chance")

    # Résumé rapide : combien de scénarios et quelle intensité de bruit.
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Scénarios demandés", resultat["nb_scenarios"])
    col2.metric("Scénarios optimaux", resultat["nb_optimaux"])
    col3.metric("Sigma points", f"{100 * resultat['sigma_points']:.1f}%")
    col4.metric("Sigma coûts", f"{100 * resultat['sigma_couts']:.1f}%")

    comptage_df = resultat["comptage_df"].copy()
    scenarios_df = resultat["scenarios_df"].copy()
    coureurs_base = set()
    if resultat_base:
        for equipe in resultat_base.get("equipes", {}).values():
            coureurs_base.update(equipe)

    metric_labels = {
        "choisi_total": "Comptage total",
        "freq_total_par_slot": "Fréquence totale",
        "choisi_p0": "Comptage période 0",
        "freq_p0": "Fréquence période 0",
        "choisi_p10": "Comptage période 10",
        "freq_p10": "Fréquence période 10",
        "choisi_p15": "Comptage période 15",
        "freq_p15": "Fréquence période 15",
    }

    tab_top, tab_table, tab_scenarios = st.tabs(["Classement", "Comptage complet", "Scénarios"])

    with tab_top:
        # L'utilisateur choisit la mesure qu'il veut regarder.
        metric = st.selectbox(
            "Mesure à afficher",
            [
                "choisi_total",
                "freq_total_par_slot",
                "choisi_p0",
                "freq_p0",
                "choisi_p10",
                "freq_p10",
                "choisi_p15",
                "freq_p15",
            ],
            index=7,
            format_func=lambda key: metric_labels[key],
        )
        top_n = st.slider("Nombre de coureurs affichés", min_value=10, max_value=80, value=30, step=5)
        top = comptage_df.sort_values(metric, ascending=False).head(top_n).copy()
        top = top.sort_values(metric, ascending=True)

        # Barres horizontales : plus lisible pour des noms de coureurs.
        couleurs = ["#d62728" if coureur in coureurs_base else "#2e86ab" for coureur in top["coureur"]]
        fig_height = max(6, top_n * 0.28)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.barh(top["coureur"], top[metric], color=couleurs, edgecolor="black", linewidth=0.5)
        ax.set_xlabel(metric_labels[metric])
        ax.set_ylabel("Coureurs")
        ax.set_title("Coureurs les plus souvent sélectionnés en simulation")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        if coureurs_base:
            st.caption("Rouge : coureur présent dans l'équipe optimale de base. Bleu : coureur non présent dans l'équipe optimale de base.")

        st.markdown("### Comprendre un coureur bleu")
        candidats = comptage_df["coureur"].tolist()
        coureur_defaut = "Paret- Peintre Valentin"
        if coureur_defaut in candidats:
            index_defaut = candidats.index(coureur_defaut)
        else:
            candidats_bleus = [p for p in candidats if p not in coureurs_base]
            index_defaut = candidats.index(candidats_bleus[0]) if candidats_bleus else 0

        coureur_analyse = st.selectbox(
            "Coureur à analyser",
            candidats,
            index=index_defaut,
        )
        if st.button("Analyser pourquoi il n'est pas dans l'équipe de base"):
            st.session_state["coureur_analyse_simulation"] = coureur_analyse

        if st.session_state.get("coureur_analyse_simulation"):
            afficher_analyse_coureur_force(
                resultat_base,
                st.session_state["coureur_analyse_simulation"],
                int(budget_initial),
                int(taille_equipe),
            )

    with tab_table:
        # Table complète, pas seulement le top.
        comptage_affichage = comptage_df.rename(
            columns={
                "coureur": "Coureur",
                "prix_base": "Prix base",
                "points_base": "Points base",
                "choisi_total": "Choisi total",
                "choisi_au_moins_une_periode": "Choisi au moins une période",
                "choisi_p0": "Choisi P0",
                "choisi_p10": "Choisi P10",
                "choisi_p15": "Choisi P15",
                "freq_total_par_slot": "Fréquence totale",
                "freq_scenario": "Fréquence scénario",
                "freq_p0": "Fréquence P0",
                "freq_p10": "Fréquence P10",
                "freq_p15": "Fréquence P15",
            }
        )
        st.dataframe(comptage_affichage, use_container_width=True, hide_index=True)
        csv_bytes = comptage_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "Télécharger comptage_simulation_chance.csv",
            data=csv_bytes,
            file_name="comptage_simulation_chance.csv",
            mime="text/csv",
        )

    with tab_scenarios:
        # Une ligne par scénario pour vérifier les statuts CBC.
        scenarios_affichage = scenarios_df.rename(
            columns={
                "scenario": "Scénario",
                "status": "Statut",
                "taille_equipe": "Taille équipe",
                "objectif": "Objectif",
            }
        )
        st.dataframe(scenarios_affichage, use_container_width=True, hide_index=True)
        if not scenarios_df.empty:
            status_counts = scenarios_df["status"].value_counts().reset_index()
            status_counts.columns = ["Statut", "Nombre"]
            st.write("Statuts solveur")
            st.dataframe(status_counts, use_container_width=True, hide_index=True)


def main():
    # Point d'entrée de l'application Streamlit.
    st.set_page_config(page_title="Fantasy Cyclisme - PuLP", layout="wide")
    st.title("Fantasy Cyclisme")

    with st.sidebar:
        # Tous les réglages sont regroupés dans la barre latérale.
        # st.header("Données")
        # st.caption("Source des données : trait_donnees.py")

        st.header("Paramètres")
        budget_initial = st.number_input("Budget initial", min_value=1, value=140, step=1)
        taille_equipe = st.number_input("Taille équipe", min_value=1, max_value=30, value=14, step=1)
        # st.caption(f"CBC fixé à {TEMPS_LIMITE_CBC} secondes, tolérance d'optimalité à {MIP_GAP_RELATIF:.0%}.")

        run_avec_transfert = st.button("Analyse classique avec transferts", type="primary")
        run_sans_transfert = st.button("Analyse sans transfert")

        st.header("Facteur Chance")
        # Paramètres de la simulation : on contrôle le nombre de tests et le niveau de hasard.
        n_scenarios = st.number_input("Nombre de scénarios", min_value=1, max_value=1500, value=20, step=10)
        sigma_points_pct = st.slider("Écart-type points (%)", min_value=0, max_value=100, value=15, step=1)
        sigma_couts_pct = st.slider("Écart-type coûts (%)", min_value=0, max_value=100, value=10, step=1)
        sigma_taille = st.number_input("Écart-type taille équipe", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        st.caption(f"Taille d'équipe simulée fixée entre {TAILLE_MIN_SIMULATION} et {TAILLE_MAX_SIMULATION}.")
        # st.caption(f"Chaque scénario est limité à {TEMPS_LIMITE_PAR_SCENARIO} secondes, avec une tolérance d'optimalité de {TOLERANCE_OPTIMALITE_SIMULATION:.0%}.")
        seed_sim = st.number_input("Seed simulation", min_value=0, max_value=1_000_000, value=42, step=1)
        run_sim = st.button("Lancer les simulations")

    # st.caption("Modèle PuLP inspiré de post_ex_lst_transferts.py, avec périodes 0, 10 et 15.")

    if run_avec_transfert or run_sans_transfert:
        # Boutons principaux : on lance l'optimisation exacte avec ou sans transferts.
        autoriser_transferts = bool(run_avec_transfert)
        libelle_mode = "avec transferts" if autoriser_transferts else "sans transfert"
        with st.spinner("Résolution PuLP/CBC en cours..."):
            try:
                resultat = optimiser_equipe_pulp(
                    budget_initial=int(budget_initial),
                    taille_equipe=int(taille_equipe),
                    abandons_10=ABANDONS_10,
                    abandons_15_extra=ABANDONS_15,
                    time_limit=TEMPS_LIMITE_CBC,
                    gap_rel=MIP_GAP_RELATIF,
                    autoriser_transferts=autoriser_transferts,
                )
            except Exception as exc:
                st.exception(exc)
                return

        st.session_state["app_pulp_resultat"] = resultat
        st.session_state["app_pulp_mode"] = libelle_mode
        st.session_state.pop("sensibilite_points", None)

    if run_sim:
        # Bouton simulation : on lance plusieurs optimisations avec des données perturbées.
        # Barre de progression pendant les scénarios.
        progress = st.progress(0)
        status_box = st.empty()

        def on_progress(done, total, status):
            # Cette petite fonction est appelée après chaque scénario.
            progress.progress(done / total)
            status_box.caption(f"Scénario {done}/{total} - statut: {status}")

        with st.spinner("Simulation Monte Carlo en cours..."):
            try:
                resultat_base = optimiser_equipe_pulp(
                    budget_initial=int(budget_initial),
                    taille_equipe=int(taille_equipe),
                    abandons_10=ABANDONS_10,
                    abandons_15_extra=ABANDONS_15,
                    time_limit=TEMPS_LIMITE_CBC,
                    gap_rel=MIP_GAP_RELATIF,
                )
                simulation = simuler_facteur_chance(
                    budget_initial=int(budget_initial),
                    taille_equipe_base=int(taille_equipe),
                    abandons_10=ABANDONS_10,
                    abandons_15_extra=ABANDONS_15,
                    n_scenarios=int(n_scenarios),
                    sigma_points=float(sigma_points_pct) / 100.0,
                    sigma_couts=float(sigma_couts_pct) / 100.0,
                    sigma_taille=float(sigma_taille),
                    taille_min=TAILLE_MIN_SIMULATION,
                    taille_max=TAILLE_MAX_SIMULATION,
                    time_limit=TEMPS_LIMITE_PAR_SCENARIO,
                    gap_rel=TOLERANCE_OPTIMALITE_SIMULATION,
                    seed=int(seed_sim),
                    progress_callback=on_progress,
                )
            except Exception as exc:
                progress.empty()
                status_box.empty()
                st.exception(exc)
                return

        progress.empty()
        status_box.empty()
        st.session_state["app_pulp_resultat"] = resultat_base
        st.session_state["app_pulp_simulation"] = simulation
        st.session_state.pop("sensibilite_points", None)

    # On garde les résultats en session_state pour ne pas les perdre au moindre clic.
    if "app_pulp_resultat" in st.session_state:
        afficher_resultats(st.session_state["app_pulp_resultat"])
    elif "app_pulp_simulation" not in st.session_state:
        st.info("Choisissez les paramètres puis lance l'optimisation !")

    # La simulation peut être affichée en plus de l'optimisation classique.
    if "app_pulp_simulation" in st.session_state:
        st.divider()
        afficher_simulation_chance(
            st.session_state["app_pulp_simulation"],
            st.session_state.get("app_pulp_resultat"),
            int(budget_initial),
            int(taille_equipe),
        )
    else:
        st.caption("La simulation chance n'a pas encore été lancée.")


if __name__ == "__main__":
    main()
