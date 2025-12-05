//Yuxiang ZHANG 21202829
//Antoine Lecomte 21103457

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Reseau.h"
#include "Hachage.h"
#include "ArbreQuat.h"
#include "Graphe.h"

int main(int argc, char *argv[]){
    if (argc != 3){
        printf("Usage: %s <fichier.cha> <methode>\n", argv[0]);
        printf("Methode :\n");
        printf("  1 : Liste\n");
        printf("  2 : Table de hachage\n");
        printf("  3 : Arbre\n");
        return 1;
    }
    FILE *fichier_cha = fopen(argv[1], "r");
    if (fichier_cha == NULL){
        printf("Erreur lors de l'ouverture du fichier\n");
        return 1;
    }
    int methode = atoi(argv[2]);
    Chaines *C = lectureChaines(fichier_cha);
    fclose(fichier_cha);
    if (C == NULL){
        printf("Erreur lors de la lecture du fichier %s\n", argv[1]);
        return 1;
    }
    Reseau* R = NULL;
    switch (methode){
        case 1:
            R = reconstitueReseauListe(C);
            break;
        case 2:
            R = reconstitueReseauHachage(C, 100);
            break;
        case 3:
            R = reconstitueReseauArbre(C);
            break;
        default:
            printf("Methode non reconnue\n");
            return 1;
    }
    ecrireReseau(R, stdout);
    // Appel de reorganiseReseau pour réorganiser le réseau (exercice 7)
    int res = reorganiseReseau(R);
    if(res == 1) {
        printf("\nLe réseau respecte la contrainte gamma.\n");
    } else {
        printf("\nLe réseau ne respecte pas la contrainte gamma.\n");
    }

    libererReseau(R);
    return 0;
}
