import matplotlib.pyplot as plt
import numpy as np

# Données pour les classifications binaires
binary_data = {
    "Perceptron (df2array)": {
        "Temps": 1.2524452209472656,
        "Train Accuracy": 0.9884393063583815,
        "Test Accuracy": 0.8220663265306123,
        "Mean Accuracy": 0.9882,
        "Std Dev": 0.0144
    },
    "Perceptron (Bag-of-Words Binaire)": {
        "Temps": 0.6399490833282471,
        "Train Accuracy": 0.9884393063583815,
        "Test Accuracy": 0.8233418367346939,
        "Mean Accuracy": 0.9882,
        "Std Dev": 0.0144
    },
    "Perceptron (Bag-of-Words Count)": {
        "Temps": 0.6322243213653564,
        "Train Accuracy": 0.9884393063583815,
        "Test Accuracy": 0.8418367346938775,
        "Mean Accuracy": 0.9765,
        "Std Dev": 0.0118
    },
    "Perceptron (Bag-of-Words Freq)": {
        "Temps": 0.6305558681488037,
        "Train Accuracy": 0.9884393063583815,
        "Test Accuracy": 0.8392857142857143,
        "Mean Accuracy": 0.8176,
        "Std Dev": 0.0343
    },
    "Perceptron (TF-IDF)": {
        "Temps": 0.6359288692474365,
        "Train Accuracy": 0.9884393063583815,
        "Test Accuracy": 0.8367346938775511,
        "Mean Accuracy": 0.8353,
        "Std Dev": 0.0546
    },
    "PerceptronBiais (df2array)": {
        "Temps": 1.2888462543487549,
        "Train Accuracy": 0.9942196531791907,
        "Test Accuracy": 0.8647959183673469,
        "Mean Accuracy": 0.9882,
        "Std Dev": 0.0144
    },
    "PerceptronBiais (Bag-of-Words Binaire)": {
        "Temps": 0.6842498779296875,
        "Train Accuracy": 0.9942196531791907,
        "Test Accuracy": 0.8628826530612245,
        "Mean Accuracy": 0.9882,
        "Std Dev": 0.0144
    },
    "PerceptronBiais (Bag-of-Words Count)": {
        "Temps": 0.7144351005554199,
        "Train Accuracy": 0.9942196531791907,
        "Test Accuracy": 0.8654336734693877,
        "Mean Accuracy": 0.9882,
        "Std Dev": 0.0144
    },
    "PerceptronBiais (Bag-of-Words Freq)": {
        "Temps": 0.7543280124664307,
        "Train Accuracy": 0.884393063583815,
        "Test Accuracy": 0.7110969387755102,
        "Mean Accuracy": 0.9824,
        "Std Dev": 0.0144
    },
    "PerceptronBiais (TF-IDF)": {
        "Temps": 0.7282757759094238,
        "Train Accuracy": 0.9942196531791907,
        "Test Accuracy": 0.9177295918367347,
        "Mean Accuracy": 0.9882,
        "Std Dev": 0.0144
    },
    "KNN k=3 (Euclidean, Binaire)": {
        "Temps": 2.79603,
        "Train Accuracy": 0.9133,
        "Test Accuracy": 0.5236,
        "Mean Accuracy": 0.5588,
        "Std Dev": 0.0744
    },
    "KNN k=3 (Euclidean, Comptage)": {
        "Temps": 2.79690,
        "Train Accuracy": 0.9306,
        "Test Accuracy": 0.6129,
        "Mean Accuracy": 0.5647,
        "Std Dev": 0.0937
    },
    "KNN k=3 (Euclidean, Frequence)": {
        "Temps": 2.74317,
        "Train Accuracy": 0.9827,
        "Test Accuracy": 0.7577,
        "Mean Accuracy": 0.7059,
        "Std Dev": 0.0588
    },
    "KNN k=3 (Euclidean, TF-IDF)": {
        "Temps": 2.80374,
        "Train Accuracy": 0.9942,
        "Test Accuracy": 0.8361,
        "Mean Accuracy": 0.7882,
        "Std Dev": 0.0861
    },
    "KNN k=3 (Cosinus, Binaire)": {
        "Temps": 1.33372,
        "Train Accuracy": 0.9191,
        "Test Accuracy": 0.8323,
        "Mean Accuracy": 0.8647,
        "Std Dev": 0.0478
    },
    "KNN k=3 (Cosinus, Comptage)": {
        "Temps": 1.31408,
        "Train Accuracy": 0.9133,
        "Test Accuracy": 0.8629,
        "Mean Accuracy": 0.8529,
        "Std Dev": 0.0322
    },
    "KNN k=3 (Cosinus, Frequence)": {
        "Temps": 1.28902,
        "Train Accuracy": 0.9133,
        "Test Accuracy": 0.8622,
        "Mean Accuracy": 0.8529,
        "Std Dev": 0.0322
    },
    "KNN k=3 (Cosinus, TF-IDF)": {
        "Temps": 1.44204,
        "Train Accuracy": 0.9480,
        "Test Accuracy": 0.8807,
        "Mean Accuracy": 0.8941,
        "Std Dev": 0.0399
    },
    "Naive Bayes (Bag-of-Words Binaire) - Version 1": {
        "Temps": 0.0023260116577148438,
        "Train Accuracy": 0.9826589595375722,
        "Test Accuracy": 0.920280612244898,
        "Mean Accuracy": 0.5035,
        "Std Dev": 0.0062
    },
    "Naive Bayes (Bag-of-Words Count) - Version 2": {
        "Temps": 0.0003299713134765625,
        "Train Accuracy": 0.9826589595375722,
        "Test Accuracy": 0.9196428571428571,
        "Mean Accuracy": 0.5069,
        "Std Dev": 0.0022
    },
    "Naive Bayes (Bag-of-Words Freq) - Version 3": {
        "Temps": 0.00034308433532714844,
        "Train Accuracy": 0.9884393063583815,
        "Test Accuracy": 0.8131377551020408,
        "Mean Accuracy": 0.5277,
        "Std Dev": 0.0044
    },
    "Naive Bayes (TF-IDF) - Version 4": {
        "Temps": 0.00033593177795410156,
        "Train Accuracy": 0.9884393063583815,
        "Test Accuracy": 0.8679846938775511,
        "Mean Accuracy": 0.5194,
        "Std Dev": 0.0071
    },
    "Arbre de Decision (Bag-of-Words Binaire) - Version 1": {
        "Temps": 0.016485214233398438,
        "Train Accuracy": 0.9942,
        "Test Accuracy": 0.7838,
        "Mean Accuracy": 0.7339,
        "Std Dev": 0.0694
    },
    "Arbre de Decision (Bag-of-Words Comptage) - Version 2": {
        "Temps": 0.013701915740966797,
        "Train Accuracy": 0.9942,
        "Test Accuracy": 0.7838,
        "Mean Accuracy": 0.7397,
        "Std Dev": 0.0663
    },
    "Arbre de Decision (Bag-of-Words Freq) - Version 3": {
        "Temps": 0.014307022094726562,
        "Train Accuracy": 0.9942,
        "Test Accuracy": 0.7640,
        "Mean Accuracy": 0.7049,
        "Std Dev": 0.0479
    },
    "Arbre de Decision (TF-IDF) - Version 4": {
        "Temps": 0.016049861907958984,
        "Train Accuracy": 0.9942,
        "Test Accuracy": 0.7615,
        "Mean Accuracy": 0.6877,
        "Std Dev": 0.0784
    }
}

# Données pour les classifications multi-classe
multi_class_data = {
    "Perceptron (Bag-of-Words Binaire) - Version 1": {
        "Temps": 1.2713229656219482,
        "Train Accuracy": 0.9875,
        "Test Accuracy": 0.4583,
        "Mean Accuracy": 0.4367,
        "Std Dev": 0.0223
    },
    "Perceptron (Bag-of-Words Comptage) - Version 2": {
        "Temps": 1.647778034210205,
        "Train Accuracy": 0.9875,
        "Test Accuracy": 0.4396,
        "Mean Accuracy": 0.4139,
        "Std Dev": 0.0095
    },
    "Perceptron (Bag-of-Words Frequence) - Version 3": {
        "Temps": 1.660815954208374,
        "Train Accuracy": 0.9875,
        "Test Accuracy": 0.4317,
        "Mean Accuracy": 0.4000,
        "Std Dev": 0.0188
    },
    "Perceptron (TF-IDF) - Version 4": {
        "Temps": 1.1476762294769287,
        "Train Accuracy": 0.9875,
        "Test Accuracy": 0.4479,
        "Mean Accuracy": 0.4072,
        "Std Dev": 0.0200
    },
    "PerceptronBiais (Bag-of-Words Binaire) - Version 1": {
        "Temps": 14.851285934448242,
        "Train Accuracy": 0.9891,
        "Test Accuracy": 0.4260,
        "Mean Accuracy": 0.4133,
        "Std Dev": 0.0296
    },
    "PerceptronBiais (Bag-of-Words Comptage) - Version 2": {
        "Temps": 15.709067106246948,
        "Train Accuracy": 0.9891,
        "Test Accuracy": 0.4524,
        "Mean Accuracy": 0.4256,
        "Std Dev": 0.0175
    },
    "PerceptronBiais (Bag-of-Words Frequence) - Version 3": {
        "Temps": 22.410446166992188,
        "Train Accuracy": 0.9815,
        "Test Accuracy": 0.5198,
        "Mean Accuracy": 0.5044,
        "Std Dev": 0.0120
    },
    "PerceptronBiais (TF-IDF) - Version 4": {
        "Temps": 24.897457122802734,
        "Train Accuracy": 0.9788,
        "Test Accuracy": 0.5675,
        "Mean Accuracy": 0.5583,
        "Std Dev": 0.0175
    },
    "KNN k=3 (Bag-of-Words Binaire) - Version 1 (Euclidienne)": {
        "Temps": 0.007567167282104492,
        "Train Accuracy": 0.3252,
        "Test Accuracy": 0.0783,
        "Mean Accuracy": 0.0773,
        "Std Dev": 0.0053
    },
    "KNN k=3 (Bag-of-Words Comptage) - Version 2 (Euclidienne)": {
        "Temps": 0.01612067222595215,
        "Train Accuracy": 0.3883,
        "Test Accuracy": 0.1278,
        "Mean Accuracy": 0.1182,
        "Std Dev": 0.0173
    },
    "KNN k=3 (Bag-of-Words Frequence) - Version 3 (Euclidienne)": {
        "Temps": 0.01175999641418457,
        "Train Accuracy": 0.3617,
        "Test Accuracy": 0.1158,
        "Mean Accuracy": 0.1073,
        "Std Dev": 0.0163
    },
    "KNN k=3 (TF-IDF) - Version 4 (Euclidienne)": {
        "Temps": 0.008476972579956055,
        "Train Accuracy": 0.3203,
        "Test Accuracy": 0.0981,
        "Mean Accuracy": 0.0839,
        "Std Dev": 0.0053
    },
    "KNN k=3 (Bag-of-Words Binaire) - Version 1 (Cosinus)": {
        "Temps": 0.003222942352294922,
        "Train Accuracy": 0.5501,
        "Test Accuracy": 0.2569,
        "Mean Accuracy": 0.2511,
        "Std Dev": 0.0282
    },
    "KNN k=3 (Bag-of-Words Comptage) - Version 2 (Cosinus)": {
        "Temps": 0.002089977264404297,
        "Train Accuracy": 0.5561,
        "Test Accuracy": 0.3004,
        "Mean Accuracy": 0.2761,
        "Std Dev": 0.0085
    },
    "KNN k=3 (Bag-of-Words Frequence) - Version 3 (Cosinus)": {
        "Temps": 0.0019378662109375,
        "Train Accuracy": 0.5561,
        "Test Accuracy": 0.3004,
        "Mean Accuracy": 0.2761,
        "Std Dev": 0.0085
    },
    "KNN k=3 (TF-IDF) - Version 4 (Cosinus)": {
        "Temps": 0.0018210411071777344,
        "Train Accuracy": 0.6389,
        "Test Accuracy": 0.3996,
        "Mean Accuracy": 0.3736,
        "Std Dev": 0.0226
    },
    "Naive Bayes (Bag-of-Words Binaire) - Version 1 (Multi-classe)": {
        "Temps": 0.007356166839599609,
        "Train Accuracy": 0.8867102396514162,
        "Test Accuracy": 0.4631990378833434,
        "Mean Accuracy": 0.0516,
        "Std Dev": 0.0003
    },
    "Naive Bayes (Bag-of-Words Comptage) - Version 2 (Multi-classe)": {
        "Temps": 0.0053441524505615234,
        "Train Accuracy": 0.829520697167756,
        "Test Accuracy": 0.4488274203247144,
        "Mean Accuracy": 0.0507,
        "Std Dev": 0.0005
    },
    "Naive Bayes (Bag-of-Words Frequence) - Version 3 (Multi-classe)": {
        "Temps": 0.008103132247924805,
        "Train Accuracy": 0.789760348583878,
        "Test Accuracy": 0.4141912206855081,
        "Mean Accuracy": 0.0522,
        "Std Dev": 0.0001
    },
    "Naive Bayes (TF-IDF) - Version 4 (Multi-classe)": {
        "Temps": 0.005766153335571289,
        "Train Accuracy": 0.9183006535947712,
        "Test Accuracy": 0.49284425736620563,
        "Mean Accuracy": 0.0519,
        "Std Dev": 0.0001
    },
    "Arbre de Decision (Bag-of-Words Binaire) - Version 1 (Multi-classe)": {
        "Temps": 1.058288335800171,
        "Train Accuracy": 0.9891,
        "Test Accuracy": 0.2162,
        "Mean Accuracy": 0.2037,
        "Std Dev": 0.0096
    },
    "Arbre de Decision (Bag-of-Words Comptage) - Version 2 (Multi-classe)": {
        "Temps": 1.03580904006958,
        "Train Accuracy": 0.9891,
        "Test Accuracy": 0.2191,
        "Mean Accuracy": 0.2157,
        "Std Dev": 0.0175
    },
    "Arbre de Decision (Bag-of-Words Frequence) - Version 3 (Multi-classe)": {
        "Temps": 1.2291460037231445,
        "Train Accuracy": 0.9891,
        "Test Accuracy": 0.2162,
        "Mean Accuracy": 0.2092,
        "Std Dev": 0.0124
    },
    "Arbre de Decision (TF-IDF) - Version 4 (Multi-classe)": {
        "Temps": 1.1658222675323486,
        "Train Accuracy": 0.9891,
        "Test Accuracy": 0.2163,
        "Mean Accuracy": 0.2092,
        "Std Dev": 0.0124
    }
}

# Fonction pour créer les graphes
def plot_comparison(data, title, ylabel, key):
    plt.figure(figsize=(12, 8))
    labels = []
    values = []

    for label, d in data.items():
        value = d[key]
        if value is not None:  # Ignorer les valeurs None
            labels.append(label)
            values.append(value)

    bars = plt.bar(labels, values)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel(ylabel)

    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Graphes pour les classifications binaires
plot_comparison(binary_data, "Temps d'exécution pour les classifications binaires", "Temps (secondes)", "Temps")
plot_comparison(binary_data, "Accuracy de train pour les classifications binaires", "Accuracy", "Train Accuracy")
plot_comparison(binary_data, "Accuracy de test pour les classifications binaires", "Accuracy", "Test Accuracy")
plot_comparison(binary_data, "Taux moyen de bonne classification pour les classifications binaires", "Taux moyen", "Mean Accuracy")
plot_comparison(binary_data, "Écarts-type pour les classifications binaires", "Écart-type", "Std Dev")

# Graphes pour les classifications multi-classe
plot_comparison(multi_class_data, "Temps d'exécution pour les classifications multi-classe", "Temps (secondes)", "Temps")
plot_comparison(multi_class_data, "Accuracy de train pour les classifications multi-classe", "Accuracy", "Train Accuracy")
plot_comparison(multi_class_data, "Accuracy de test pour les classifications multi-classe", "Accuracy", "Test Accuracy")
plot_comparison(multi_class_data, "Taux moyen de bonne classification pour les classifications multi-classe", "Taux moyen", "Mean Accuracy")
plot_comparison(multi_class_data, "Écarts-type pour les classifications multi-classe", "Écart-type", "Std Dev")
