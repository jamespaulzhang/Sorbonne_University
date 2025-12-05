# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2024-2025, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2025

# Import de packages externes
import numpy as np
import pandas as pd
import copy

# ---------------------------
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        predictions = np.array([self.predict(x) for x in desc_set])
        correct_predictions = np.sum(predictions == label_set)
        return correct_predictions / len(label_set)

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        Classifier.__init__(self,input_dimension)
        self.k = k
    
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set
    
    def score(self, x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        distances = np.linalg.norm(self.desc_set - x, axis=1)
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.label_set[k_nearest_indices]
        p = np.mean(k_nearest_labels == +1)
        return p
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        return 1 if self.score(x) >= 0.5 else -1
    
class ClassifierLineaireRandom(Classifier):
    """ Classifieur linéaire avec un vecteur de poids aléatoire.
    """
    def __init__(self, input_dimension):
        """ Constructeur
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
        """
        super().__init__(input_dimension)
        self.w = np.random.rand(input_dimension) * 2 - 1  # Poids initiaux aléatoires entre -1 et 1

    def train(self, desc_set, label_set):
        """ Entraînement : ici, pas d'apprentissage réel, les poids sont fixés aléatoirement """
        pass  # Aucun ajustement des poids

    def score(self, x):
        """ Calcule le score de prédiction pour x
            x: un exemple (ndarray)
            Retourne une valeur réelle correspondant au produit scalaire avec les poids
        """
        return np.dot(self.w, x)

    def predict(self, x):
        """ Prédit la classe de x
            Retourne +1 si le score est positif, -1 sinon
        """
        return 1 if self.score(x) >= 0 else -1

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        Classifier.__init__(self, input_dimension)
        self.learning_rate = learning_rate
        
        if init:
            self.w = np.zeros(input_dimension)
        else:
            self.w = (np.random.rand(input_dimension) * 2 - 1) * 0.001
        
        self.allw = [self.w.copy()]  # Stocker les poids initiaux

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        indices = list(range(len(desc_set)))
        np.random.shuffle(indices)
        
        for i in indices:
            x_i = desc_set[i]
            y_i = label_set[i]
            y_pred = np.dot(self.w, x_i)
            y_pred_sign = np.sign(y_pred)
            
            if y_pred_sign != y_i:
                self.w += self.learning_rate * y_i * x_i
                self.allw.append(self.w.copy())  # Stocker les poids après chaque mise à jour

    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """        
        differences = []
        
        for _ in range(nb_max):
            old_w = self.w.copy()
            self.train_step(desc_set, label_set)
                    
            diff = np.linalg.norm(self.w - old_w)
            differences.append(diff)
            
            if diff < seuil:  # Arrêt si convergence
                break
        
        return differences
    
    def score(self, x):
        """ Rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)
    
    def predict(self, x):
        """ Rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.sign(self.score(x))
    
    def get_allw(self):
        """ Récupère l'historique des poids """
        return self.allw

class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        # print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """  
        indices = list(range(len(desc_set)))
        np.random.shuffle(indices)
        
        for i in indices:
            x_i = desc_set[i]
            y_i = label_set[i]
            f_xi = np.dot(self.w, x_i)  # Score du perceptron
            
            if f_xi * y_i < 1:  # Critère modifié
                self.w += self.learning_rate * (y_i - f_xi) * x_i
                self.allw.append(self.w.copy())  # Stocker l'évolution des poids   
                
class ClassifierMultiOAA(Classifier):
    """ Classifieur multi-classes
    """
    def __init__(self, cl_bin):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - cl_bin: classifieur binaire positif/négatif
            Hypothèse : input_dimension > 0
        """
        self.cl_bin = cl_bin
        self.models = {}
        self.classes = None
        
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.classes = np.unique(label_set)
        
        for c in self.classes:
            # Créer une version binaire des labels : +1 pour la classe c, -1 pour les autres
            labels_bin = np.where(label_set == c, 1, -1)
            
            # Créer une copie du classifieur binaire et l'entraîner
            cl_bin_copy = copy.deepcopy(self.cl_bin)
            cl_bin_copy.train(desc_set, labels_bin)  # Entraînement sur les données
            self.models[c] = cl_bin_copy  # Stocker le classifieur pour la classe c
        
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return {c: model.score(x) for c, model in self.models.items()}  # Calculer le score du classifieur pour la classe c
        
        
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return max(self.models.keys(), key=lambda c: self.models[c].score(x))  # Retourner la classe ayant le score maximal
