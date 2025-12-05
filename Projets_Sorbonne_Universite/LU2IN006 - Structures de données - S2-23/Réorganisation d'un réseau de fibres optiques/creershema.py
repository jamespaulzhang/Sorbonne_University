# Yuxiang ZHANG 21202829
# Antoine Lecomte 21103457

import matplotlib.pyplot as plt

# Données fournies
nbPointsChaine = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
tempsListeChainee = [4.482, 19.484, 43.837, 90.311, 186.342, 307.53, 484.483, 683.438]
tempsTableHachage_1 = [4.876, 19.8, 43.698, 137.8, 342.253, 530.096, 751.494, 1091.373]
tempsTableHachage_10 = [1.311, 5.394, 12.655, 29.396, 69.522, 133.891, 222.492, 751.494]
tempsTableHachage_100 = [0.17, 0.68, 1.693, 3.651, 7.846, 14.684, 23.599, 23.599]
tempsTableHachage_1000 = [0.045, 0.13, 0.271, 0.516, 0.984, 1.725, 2.739, 2.739]
tempsArbreQuaternaire = [42.693, 173.849, 431.918, 897.147, 2110.399, 3385.756, 5022.403, None]

# Construction des graphiques
plt.figure(figsize=(18, 5))

# Graphique pour la méthode de la liste chaînée
plt.subplot(1, 3, 1)
plt.plot(nbPointsChaine, tempsListeChainee, marker='o', label='Liste chaînée', color='blue')
plt.title('Temps de calcul avec la liste chaînée')
plt.xlabel('Nombre de points total des chaînes')
plt.ylabel('Temps de calcul (secondes)')
plt.grid(True)
plt.legend()

# Graphique pour la méthode de la table de hachage
plt.subplot(1, 3, 2)
plt.plot(nbPointsChaine, tempsTableHachage_1, marker='o', label='Table de hachage (taille 1)')
plt.plot(nbPointsChaine, tempsTableHachage_10, marker='o', label='Table de hachage (taille 10)')
plt.plot(nbPointsChaine, tempsTableHachage_100, marker='o', label='Table de hachage (taille 100)')
plt.plot(nbPointsChaine, tempsTableHachage_1000, marker='o', label='Table de hachage (taille 1000)')
plt.title('Temps de calcul avec table de hachage')
plt.xlabel('Nombre de points total des chaînes')
plt.ylabel('Temps de calcul (secondes)')
plt.grid(True)
plt.legend()

# Graphique pour la méthode de l'arbre quaternaire
plt.subplot(1, 3, 3)
plt.plot(nbPointsChaine, tempsArbreQuaternaire, marker='o', label='Arbre quaternaire', color='red')
plt.title('Temps de calcul avec arbre quaternaire')
plt.xlabel('Nombre de points total des chaînes')
plt.ylabel('Temps de calcul (secondes)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()