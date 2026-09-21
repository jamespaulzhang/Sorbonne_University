#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

// Fonction pour allouer une matrice carrée de taille n x n
int** alloue_matrice(int n) {
    int** matrice = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        matrice[i] = (int*)malloc(n * sizeof(int));
    }
    return matrice;
}

// Fonction pour désallouer une matrice carrée
void desalloue_matrice(int** matrice, int n) {
    for (int i = 0; i < n; i++) {
        free(matrice[i]);
    }
    free(matrice);
}

// Fonction pour remplir une matrice carrée avec des valeurs aléatoires entre 0 et V (non inclus)
void remplir_matrice(int** matrice, int n, int V) {
    srand(time(NULL));  // Initialisation du générateur de nombres aléatoires
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrice[i][j] = rand() % V;
        }
    }
}

// Fonction pour afficher les valeurs d'une matrice carrée
void afficher_matrice(int** matrice, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", matrice[i][j]);
        }
        printf("\n");
    }
}

// Algorithme O(n^4) pour vérifier si tous les éléments d'une matrice sont distincts
bool elements_distincts_O_n4(int** matrice, int n) {
    for (int i1 = 0; i1 < n; i1++) {
        for (int j1 = 0; j1 < n; j1++) {
            for (int i2 = 0; i2 < n; i2++) {
                for (int j2 = 0; j2 < n; j2++) {
                    if ((i1 != i2 || j1 != j2) && matrice[i1][j1] == matrice[i2][j2]) {
                        return false; // Il y a des éléments égaux
                    }
                }
            }
        }
    }
    return true; // Tous les éléments sont distincts
}

// Algorithme amélioré pour vérifier si tous les éléments d'une matrice sont distincts
bool elements_distincts_ameliore(int** matrice, int n, int V) {
    int* valeurs_presentes = (int*)calloc(V, sizeof(int));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int valeur = matrice[i][j];

            // Vérifier si la valeur a déjà été rencontrée
            if (valeurs_presentes[valeur] == 1) {
                free(valeurs_presentes);
                return false; // La valeur est déjà présente
            }

            valeurs_presentes[valeur] = 1; // Marquer la valeur comme présente
        }
    }

    free(valeurs_presentes);
    return true; // Tous les éléments sont distincts
}

// Algorithme O(n^3) pour le produit de deux matrices
void produit_matrices_O_n3(int** matrice1, int** matrice2, int** resultat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            resultat[i][j] = 0;
            for (int k = 0; k < n; k++) {
                resultat[i][j] += matrice1[i][k] * matrice2[k][j];
            }
        }
    }
}

// Algorithme pour le produit de deux matrices triangulaires
void produit_matrices_triangulaires(int** matrice_sup, int** matrice_inf, int** resultat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            resultat[i][j] = 0;
            for (int k = i; k < n; k++) {
                resultat[i][j] += matrice_sup[i][k] * matrice_inf[k][j];
            }
        }
    }
}


int main() {
    // Exemple d'utilisation
    int n = 3;  // Taille de la matrice
    int V = 10; // Borne maximale pour les valeurs aléatoires

    // Allouer, remplir et afficher une matrice carrée
    int** matrice = alloue_matrice(n);
    remplir_matrice(matrice, n, V);
    afficher_matrice(matrice, n);

    // Libérer la mémoire
    desalloue_matrice(matrice, n);

    return 0;
}
