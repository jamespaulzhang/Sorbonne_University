//Yuxiang ZHANG 21202829
//Antoine Lecomte 21103457

#ifndef __GRAPHE_H__
#define __GRAPHE_H__

#include <stdlib.h>
#include <stdio.h>
#include "Struct_File.h"
#include "Reseau.h"

typedef struct {
    int u, v; /* Numeros des sommets extremite */
} Arete;

typedef struct cellule_arete {
    Arete* a; /* pointeur sur l’arete */
    struct cellule_arete* suiv;
} Cellule_arete;

typedef struct {
    int num; /* Numero du sommet (le meme que dans T_som) */
    double x, y;
    Cellule_arete* L_voisin; /* Liste chainee des voisins */
} Sommet;

typedef struct {
    int e1, e2; /* Les deux extremites de la commodite */
} Commod;

typedef struct {
    int nbsom; /* Nombre de sommets */
    Sommet** T_som; /* Tableau de pointeurs sur sommets */
    int gamma;
    int nbcommod; /* Nombre de commodites */
    Commod* T_commod; /* Tableau des commodites */
} Graphe;

Graphe* creerGraphe(Reseau* r);
int MinNbAretes(Graphe* g, int u, int v);
S_file* MinListeEntiers(Graphe* g, int u, int v);
int reorganiseReseau(Reseau* r);

int verifierNbChaines(Graphe* g, int** matrice);
void libererGraphe(Graphe* g);
void libererFile(S_file* f);

#endif