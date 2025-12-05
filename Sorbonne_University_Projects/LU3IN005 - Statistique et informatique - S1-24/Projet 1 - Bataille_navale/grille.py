# Sam ASLO 21210657
# Yuxiang ZHANG 21202829

import numpy as np


class Grille:
    def __init__(self, taille):
        self.taille = taille
        self.cases = np.zeros((taille, taille), dtype=int)
