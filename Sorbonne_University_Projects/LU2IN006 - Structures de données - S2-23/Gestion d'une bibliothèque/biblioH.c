#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "biblioH.h"

// Fonction de hachage pour calculer la clé à partir de l'auteur
int fonctionClef(char* auteur){
    int clef = 0;
    // Somme des valeurs ASCII des caractères de l'auteur
    while (*auteur != '\0'){
        clef += (int)(*auteur);
        auteur++;
    }
    return clef;
}

// Fonction pour créer un nouveau livre dans une table de hachage
LivreH* creer_livreH(int num, char* titre, char* auteur){
    LivreH* l = (LivreH*)malloc(sizeof(LivreH));
    if (l == NULL){
        fprintf(stderr, "Erreur lors de la création d'un livre");
        exit(EXIT_FAILURE);
    }
    l->clef = fonctionClef(auteur);
    l->num = num;
    l->titre = strdup(titre);
    l->auteur = strdup(auteur);
    l->suivant = NULL;
    return l;
}

// Fonction pour libérer la mémoire occupée par un livre
void liberer_livreH(LivreH* l){
    if (l != NULL){
        free(l->titre);
        free(l->auteur);
        free(l);
    }
}

// Fonction pour créer une nouvelle bibliothèque avec une table de hachage de taille m
BiblioH* creer_biblioH(int m){
    BiblioH* b = (BiblioH*)malloc(sizeof(BiblioH));
    if (b == NULL){
        fprintf(stderr, "Erreur lors de la création de la bibliothèque");
        exit(EXIT_FAILURE);
    }
    b->nE = 0;
    b->m = m;
    b->T = (LivreH**)malloc(sizeof(LivreH*)*m);
    if (b->T == NULL){
        fprintf(stderr, "Erreur lors de la création de la table de hachage");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < m; i++){
        b->T[i] = NULL;
    }
    return b;
}

// Fonction pour libérer la mémoire occupée par une bibliothèque
void liberer_biblioH(BiblioH* b){
    if (b != NULL) {
        for (int i = 0; i < b->m; i++){
            LivreH* courant = b->T[i];
            while (courant != NULL){
                LivreH* suivant = courant->suivant;
                liberer_livreH(courant);
                courant = suivant;
            }
        }
        free(b->T);
        free(b);
    }
}

// Fonction de hachage pour calculer l'indice dans la table de hachage
int fonctionHachage(int cle, int m){
    const double A = (sqrt(5)-1)/2;
    int res = (int)(m*((cle*A)-(int)(cle*A)));  // On récupère la partie décimale de cle*A puis on fait le produit par m de ce résultat (on obtient donc une valeur entre 0 et m-1).
    return res;
}

// Fonction pour insérer un nouveau livre dans la bibliothèque
void inserer(BiblioH* b, int num, char* titre, char* auteur){
    if (b->nE == b->m){
        printf("Table pleine, impossible de rajouter de livres.\n");
        return;
    }
    else{
        int clef = fonctionClef(auteur);
        int ind = fonctionHachage(clef, b->m);
        LivreH* nouv_l = creer_livreH(num, titre, auteur);
        // Insertion en tête de liste à l'indice calculé
        nouv_l->suivant = b->T[ind];
        b->T[ind] = nouv_l;
        b->nE++;
    }
}
