# ATTENTION : CE FICHIER NE DOIS PAS ETRE MODIFIE !
# liste des classes et fonctions de ce fichier :
#  - getNthDict (d'une ligne n de dataframe à un dictionnaire)
#  - viewData (visualisation d'un dataframe)
#  - discretizeData (discrétisation automatique)
#  - AbstractClassifier (classe abstraite pour tous les classifieurs)
#  - drawGraphHorizontal (dessin d'un graphe orienté horizontal)
#  - drawGraph (dessin d'un graphe orienté vertical)

import numpy as np

from IPython.display import Image
import pydot

import matplotlib
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for making plots with seaborn

import math  # for math


def getNthDict(df, n):
    """
    Rend un dictionnaire des données de la n-ième ligne

    Parametes
    ---------
      df : pandas.dataframe
        le `pandas.dataframe` contenant les données
      n:  int
        le numéro de la ligne
    """
    return df[n:n + 1].to_dict(orient='records')[0]


def viewData(data, kde=True):
    """
    visualisation d'un pandas.dataframe sous la forme d'histogramme (avec uu gaussian kernel densiy estimate si demandé
    et intéressant)

    Parameters
    ----------
      data : pandas.dataFrame
        le `pandas.dataframe` à visualiser
      kde : bool
        demande de l'affichage du gaussian kdf (default: True)
    """
    x = 4
    y = math.ceil(len(data.keys()) / x)
    plt.figure(figsize=(x * 4, y * 2))
    for i, k in enumerate(data.keys()):
        ax = plt.subplot(x, y, i + 1, xticklabels=[])
        ax.set_title("Distribution of '{0}': {1} in [{2},{3}]".format(
            k, len(data[k].unique()), data[k].min(), data[k].max()))
        dd = sns.histplot(data[k], kde=kde and len(
            data[k].unique()) > 5, ax=ax, color="blue")
        dd.set_title("")


def discretizeData(data):
    """
    Discrétisation automatique utilisant numpy

    Parameters
    ----------
      data : pandas.dataframe
        le `pandas.dataframe` dont certaines colonnes seront à discrétisées

    Returns
    -------
      pandas.dataframe
        le nouveau `pandas.dataframe` discréité
    """
    newData = data.copy()
    for k in newData.keys():
        if len(newData[k].unique()) > 5:
            newData[k] = data.apply(lambda row: np.digitize(row[k], np.histogram_bin_edges(newData[k], bins="fd")),
                                    axis=1)
    return newData


class AbstractClassifier:
    """
    Un classifier implémente un algorithme pour estimer la classe d'un vecteur d'attributs. Il propose aussi comme service
    de calculer les statistiques de reconnaissance à partir d'un pandas.dataframe.
    """

    def __init__(self):
        pass

    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        Parameters
         ---------
          attr: Dict[str,val] 
            Le dictionnaire nom-valeur des attributs

        Returns
        -------
          la classe 0 ou 1 estimée
        """
        raise NotImplementedError

    def statsOnDF(self, df):
        """
        à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.

        Parameters
        ----------
          df:  pandas.DataFram
            le dataframe à tester

        Returns
        -------
          Dict[str,float]
            un dictionnaire incluant les VP,FP,VN,FN,précision et rappel
        """
        raise NotImplementedError


__GRAPHPREAMBULE = 'digraph{margin="0,0";node [style=filled, color = black, fillcolor=lightgrey,fontsize=10,' \
                   'shape=box,margin=0.05,width=0,height=0];'


def drawGraphHorizontal(arcs):
    """
    Dessine un graph (horizontalement) à partir d'une chaîne décrivant ses arcs (et noeuds)  (par exemple 'A->B;C->A')"

    Parameters
    ----------
      arcs: str
        la chaîne contenant les arcs

    Returns
    -------
      Image
        l'image représentant le graphe
    """
    graph = pydot.graph_from_dot_data(
        __GRAPHPREAMBULE + 'rankdir=LR;' + arcs + '}')[0]
    return Image(graph.create_png())


def drawGraph(arcs):
    """
    Dessine un graph à partir d'une chaîne décrivant ses arcs (et noeuds)  (par exemple 'A->B;C->A')"

    Parameters
    ----------
      arcs: str
        la chaîne contenant les arcs

    Returns
    -------
      Image
        l'image représentant le graphe
    """
    graph = pydot.graph_from_dot_data(__GRAPHPREAMBULE + arcs + '}')[0]
    return Image(graph.create_png())
