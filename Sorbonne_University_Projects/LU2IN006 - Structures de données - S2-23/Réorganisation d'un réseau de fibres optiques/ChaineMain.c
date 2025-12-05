//Yuxiang ZHANG 21202829
//Antoine Lecomte 21103457

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Chaine.h"
#include "SVGwriter.h"

int main(int argc, char *argv[]){
    if (argc != 3){
        printf("Usage: %s input_file output_file\n", argv[0]);
        return 1;
    }
    FILE *input_file = fopen(argv[1], "r");
    if (input_file == NULL){
        fprintf(stderr, "Erreur lors de l'ouverture du fichier d'entrée\n");
        return 1;
    }
    Chaines *chaines = lectureChaines(input_file);
    fclose(input_file);
    if (chaines == NULL){
        fprintf(stderr, "Erreur lors de la lecture des chaînes.\n");
        return 1;
    }
    FILE *output_file = fopen(argv[2], "w");
    if (output_file == NULL){
        fprintf(stderr, "Erreur lors de l'ouverture du fichier de sortie\n");
        libererChaines(chaines);
        return 1;
    }
    ecrireChaines(chaines, output_file);
    fclose(output_file);
    afficheChainesSVG(chaines, "output.svg");
    libererChaines(chaines);
    return 0;
}
