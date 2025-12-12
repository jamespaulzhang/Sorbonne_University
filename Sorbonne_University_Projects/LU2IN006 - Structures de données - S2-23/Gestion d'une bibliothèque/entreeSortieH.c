#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "biblioH.h"

// Fonction pour charger les n premières entrées d'un fichier dans une bibliothèque à table de hachage
BiblioH* charger_n_entreesH(char* nomfic, int n) {
    FILE* fichier = fopen(nomfic, "r");
    if (fichier == NULL) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier %s.\n", nomfic);
        exit(EXIT_FAILURE);
    }
    BiblioH* biblio = creer_biblioH(100); // Taille de la table de hachage à ajuster
    char ligne[256];
    int i = 0;
    while (fgets(ligne, sizeof(ligne), fichier) && i < n) {
        int num;
        char titre[256];
        char auteur[256];
        if (sscanf(ligne, "%d %s %s", &num, titre, auteur) == 3) {
            inserer(biblio, num, titre, auteur);
            i++;
        }
        else {
            fprintf(stderr, "Erreur de format de ligne dans le fichier %s.\n", nomfic);
        }
    }
    fclose(fichier);
    return biblio;
}

// Fonction pour enregistrer une bibliothèque à table de hachage dans un fichier
void enregistrer_biblioH(BiblioH* b, char* nomfic) {
    FILE* fichier = fopen(nomfic, "w");
    if (fichier == NULL) {
        fprintf(stderr, "Erreur lors de l'ouverture ou de la création du fichier %s.\n", nomfic);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < b->m; i++) {
        LivreH* courant = b->T[i];
        while (courant != NULL) {
            fprintf(fichier, "%d %d %s %s\n", courant->clef, courant->num, courant->titre, courant->auteur);
            courant = courant->suivant;
        }
    }
    fclose(fichier);
}

// Fonction pour afficher les informations d'un livre
void afficher_livreH(LivreH* livre) {
    if (livre != NULL) {
        printf("Clé: %d, Numéro: %d, Titre: %s, Auteur: %s\n", livre->clef, livre->num, livre->titre, livre->auteur);
    }
    else {
        printf("Livre inexistant.\n");
    }
}

// Fonction pour afficher tous les livres d'une bibliothèque à table de hachage
void afficher_biblioH(BiblioH* b) {
    printf("Bibliothèque :\n");
    for (int i = 0; i < b->m; i++) {
        LivreH* courant = b->T[i];
        while (courant != NULL) {
            afficher_livreH(courant);
            courant = courant->suivant;
        }
    }
}

// Fonction pour rechercher un livre par son numéro dans une bibliothèque à table de hachage
LivreH* rechercher_par_numeroH(BiblioH* b, int num) {
    for (int i = 0; i < b->m; i++) {
        LivreH* courant = b->T[i];
        while (courant != NULL) {
            if (courant->num == num) {
                return courant;
            }
            courant = courant->suivant;
        }
    }
    return NULL;
}

// Fonction pour rechercher un livre par son titre dans une bibliothèque à table de hachage
LivreH* rechercher_par_titreH(BiblioH* b, char* titre) {
    for (int i = 0; i < b->m; i++) {
        LivreH* courant = b->T[i];
        while (courant != NULL) {
            if (strcmp(courant->titre, titre) == 0) {
                return courant;
            }
            courant = courant->suivant;
        }
    }
    return NULL;
}

// Fonction pour rechercher tous les livres d'un auteur dans une bibliothèque à table de hachage
BiblioH* rechercher_par_auteurH(BiblioH* b, char* auteur) {
    BiblioH* b_new = creer_biblioH(b->m);
    int clef = fonctionClef(auteur);
    int index = fonctionHachage(clef, b->m);
    LivreH* courant = b->T[index];
    while (courant != NULL) {
        if (strcmp(courant->auteur, auteur) == 0) {
            // Insertion du livre dans la bibliothèque des résultats
            inserer(b_new, courant->clef, courant->titre, courant->auteur);
        }
        courant = courant->suivant;
    }
    return b_new;
}

// Fonction pour supprimer un livre d'une bibliothèque à table de hachage
void supprimer_livreH(BiblioH* biblio, int num, char* titre, char* auteur) {
    // Calcul de la clé de l'auteur pour obtenir l'indice dans la table de hachage
    int clef = fonctionClef(auteur);
    int index = fonctionHachage(clef, biblio->m);
    LivreH* precedent = NULL;
    LivreH* courant = biblio->T[index];
    int trouve = 0;
    while (courant != NULL) {
        // Vérification si le livre courant correspond à celui recherché
        if (courant->num == num && strcmp(courant->titre, titre) == 0 && strcmp(courant->auteur, auteur) == 0) {
            // Suppression du livre de la liste chainée
            if (precedent != NULL) {
                precedent->suivant = courant->suivant;
            }
            else {
                biblio->T[index] = courant->suivant;
            }
            liberer_livreH(courant);
            trouve = 1;
            printf("Ouvrage supprimé.\n");
            break;
        }
        else {
            precedent = courant;
            courant = courant->suivant;
        }
    }
    if (!trouve) {
        printf("Livre non trouvé pour suppression.\n");
    }
}

// Fonction pour fusionner deux bibliothèques à table de hachage
void fusionner_bibliothequesH(BiblioH* biblio1, BiblioH* biblio2) {
    for (int i = 0; i < biblio2->m; i++) {
        LivreH* courant = biblio2->T[i];
        while (courant != NULL) {
            // Insertion du livre dans la première bibliothèque
            inserer(biblio1, courant->num, courant->titre, courant->auteur);
            LivreH* suivant = courant->suivant;
            liberer_livreH(courant);
            courant = suivant;
        }
    }
    free(biblio2->T);
    free(biblio2);
}

// Fonction pour rechercher les exemplaires multiples dans une bibliothèque à table de hachage
BiblioH* rechercher_exemplaires_multiplesH(BiblioH* biblio) {
    BiblioH* resultat = creer_biblioH(biblio->m);
    for (int i = 0; i < biblio->m; i++) {
        LivreH* courant = biblio->T[i];
        while (courant != NULL) {
            LivreH* temp = courant->suivant;
            while (temp != NULL) {
                // Vérification si les livres ont le même titre et auteur
                if (strcmp(courant->titre, temp->titre) == 0 && strcmp(courant->auteur, temp->auteur) == 0) {
                    // Vérification si les livres ne sont pas déjà présents dans la bibliothèque des résultats
                    if (rechercher_par_numeroH(resultat, courant->num) == NULL && courant->num != temp->num) {
                        // Insertion du premier livre
                        inserer(resultat, courant->num, courant->titre, courant->auteur);
                    }
                    if (rechercher_par_numeroH(resultat, temp->num) == NULL && courant->num != temp->num) {
                        // Insertion du deuxième livre
                        inserer(resultat, temp->num, temp->titre, temp->auteur);
                    }
                }
                temp = temp->suivant;
            }
            courant = courant->suivant;
        }
    }
    return resultat;
}
