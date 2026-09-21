#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "biblioLC.h"

// Fonction pour créer un nouveau livre avec les informations données
Livre* creer_livre(int num, char* titre, char* auteur){
    Livre* nouveau_livre = (Livre*)malloc(sizeof(Livre));
    if (nouveau_livre == NULL){
        fprintf(stderr, "Erreur lors de l'allocation de mémoire pour le nouveau livre.\n");
        exit(EXIT_FAILURE);
    }
    // Allocation de mémoire pour le titre et l'auteur, puis copie des chaînes de caractères
    nouveau_livre->titre = (char*)malloc(strlen(titre) + 1);
    nouveau_livre->auteur = (char*)malloc(strlen(auteur) + 1);
    if (nouveau_livre->titre == NULL || nouveau_livre->auteur == NULL){
        fprintf(stderr, "Erreur lors de l'allocation de mémoire pour le titre ou l'auteur du nouveau livre.\n");
        exit(EXIT_FAILURE);
    }
    strcpy(nouveau_livre->titre, titre);
    strcpy(nouveau_livre->auteur, auteur);
    nouveau_livre->num = num;
    nouveau_livre->suiv = NULL;
    return nouveau_livre;
}

// Fonction pour libérer la mémoire occupée par un livre
void liberer_livre(Livre* l){
    if (l != NULL){
        free(l->titre);
        free(l->auteur);
        free(l);
    }
}

// Fonction pour créer une nouvelle bibliothèque vide
Biblio* creer_biblio(){
    Biblio* nouvelle_biblio = (Biblio*)malloc(sizeof(Biblio));
    if (nouvelle_biblio == NULL){
        fprintf(stderr, "Erreur lors de l'allocation de mémoire pour la nouvelle bibliothèque.\n");
        exit(EXIT_FAILURE);
    }
    nouvelle_biblio->L = NULL;
    return nouvelle_biblio;
}

// Fonction pour libérer la mémoire occupée par une bibliothèque
void liberer_biblio(Biblio* b){
    if (b != NULL){
        Livre* courant = b->L;
        while (courant != NULL){
            Livre* temp = courant;
            courant = courant->suiv;
            liberer_livre(temp);
        }
        free(b);
    }
}

// Fonction pour insérer un nouveau livre en tête de la bibliothèque
void inserer_en_tete(Biblio* b, int num, char* titre, char* auteur){
    Livre* nouveau_livre = creer_livre(num, titre, auteur);
    nouveau_livre->suiv = b->L;
    b->L = nouveau_livre;
}
