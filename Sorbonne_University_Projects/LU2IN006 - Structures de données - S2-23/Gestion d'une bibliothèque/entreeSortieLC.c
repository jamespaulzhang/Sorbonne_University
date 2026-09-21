#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "entreeSortieLC.h"

// Fonction pour charger les n premières entrées d'un fichier dans une bibliothèque à liste chainée
Biblio* charger_n_entrees(char* nomfic, int n){
    FILE* fichier = fopen(nomfic, "r");
    if (fichier == NULL){
        fprintf(stderr, "Erreur lors de l'ouverture du fichier %s.\n", nomfic);
        exit(EXIT_FAILURE);
    }
    Biblio* biblio = creer_biblio();
    char ligne[256];
    int i = 0;
    while (fgets(ligne, sizeof(ligne), fichier) && i < n){
        int num;
        char titre[256];
        char auteur[256];
        if (sscanf(ligne, "%d %s %s", &num, titre, auteur) == 3){
            inserer_en_tete(biblio, num, titre, auteur);
            i++;
        }
        else{
            fprintf(stderr, "Erreur de format de ligne dans le fichier %s.\n", nomfic);
        }
    }
    fclose(fichier);
    return biblio;
}

// Fonction pour enregistrer une bibliothèque à liste chainée dans un fichier
void enregistrer_biblio(Biblio* b, char* nomfic){
    FILE* fichier = fopen(nomfic, "w");
    if (fichier == NULL){
        fprintf(stderr, "Erreur lors de l'ouverture ou de la création du fichier %s.\n", nomfic);
        exit(EXIT_FAILURE);
    }
    Livre* courant = b->L;
    while (courant != NULL){
        fprintf(fichier, "%d %s %s\n", courant->num, courant->titre, courant->auteur);
        courant = courant->suiv;
    }

    fclose(fichier);
}

// Fonction pour afficher les informations d'un livre
void afficher_livre(Livre* livre){
    if (livre != NULL){
        printf("Numéro: %d, Titre: %s, Auteur: %s\n", livre->num, livre->titre, livre->auteur);
    }
    else{
        printf("Livre inexistant.\n");
    }
}

// Fonction pour afficher tous les livres d'une bibliothèque à liste chainée
void afficher_biblio(Biblio* b){
    Livre* courant = b->L;
    printf("Bibliothèque :\n");
    while (courant != NULL){
        afficher_livre(courant);
        courant = courant->suiv;
    }
}

// Fonction pour rechercher un livre par son numéro dans une bibliothèque à liste chainée
Livre* rechercher_par_numero(Biblio* b, int n){
    Livre* courant = b->L;
    while (courant != NULL){
        if (courant->num == n){
            return courant;
        }
        courant = courant -> suiv;
    }
    return NULL;
}

// Fonction pour rechercher un livre par son titre dans une bibliothèque à liste chainée
Livre* rechercher_par_titre(Biblio* b, char* titre){
    Livre* courant = b->L;
    while (courant != NULL){
        if (strcmp(courant->titre, titre) == 0){
            return courant;
        }
        courant = courant->suiv;
    }
    return NULL;
}

// Fonction pour rechercher tous les livres d'un auteur dans une bibliothèque à liste chainée
Biblio* rechercher_par_auteur(Biblio* b, char* auteur){
    Biblio* b_new = creer_biblio();
    Livre* courant = b->L;
    while (courant != NULL){
        if (strcmp(courant->auteur,auteur) == 0){
            inserer_en_tete(b_new,courant->num,courant->titre,courant->auteur);
        }
        courant = courant->suiv;
    }
    return b_new;
}

// Fonction pour supprimer un livre d'une bibliothèque à liste chainée
void supprimer_livre(Biblio* biblio, int num, char* titre, char* auteur){
    Livre* precedent = NULL;
    Livre* courant = biblio->L;
    while (courant != NULL){
        // Vérification si le livre courant correspond à celui recherché
        if (courant->num == num && strcmp(courant->titre, titre) == 0 && strcmp(courant->auteur, auteur) == 0){
            // Suppression du livre de la liste chainée
            if (precedent != NULL){
                precedent->suiv = courant->suiv;
            }
            else{
                biblio->L = courant->suiv;
            }
            liberer_livre(courant);
            return;
        }
        else{
            precedent = courant;
            courant = courant->suiv;
        }
    }
    printf("Livre non trouvé pour suppression.\n");
}

// Fonction pour fusionner deux bibliothèques à liste chainée
void fusionner_bibliotheques(Biblio* biblio1, Biblio* biblio2){
    Livre* courant = biblio2->L;
    while (courant != NULL){
        inserer_en_tete(biblio1, courant->num, courant->titre, courant->auteur);
        Livre* suivant = courant->suiv;
        liberer_livre(courant);
        courant = suivant;
    }
    free(biblio2);
}

// Fonction pour rechercher les exemplaires multiples dans une bibliothèque à liste chainée
Biblio* rechercher_exemplaires_multiples(Biblio* biblio) {
    Biblio* resultat = creer_biblio();
    Livre* courant = biblio->L;
    while (courant != NULL) {
        Livre* temp = courant->suiv;
        while (temp != NULL) {
            // Vérification si les livres ont le même titre et auteur
            if (strcmp(courant->titre, temp->titre) == 0 && strcmp(courant->auteur, temp->auteur) == 0) {
                // Vérification si les livres ne sont pas déjà présents dans la bibliothèque des résultats
                if (rechercher_par_numero(resultat, courant->num) == NULL) {
                    // Insertion du premier livre
                    inserer_en_tete(resultat, courant->num, courant->titre, courant->auteur);
                }
                if (rechercher_par_numero(resultat, temp->num) == NULL) {
                    // Insertion du deuxième livre
                    inserer_en_tete(resultat, temp->num, temp->titre, temp->auteur);
                }
            }
            temp = temp->suiv;
        }
        courant = courant->suiv;
    }
    return resultat;
}
