//Yuxiang ZHANG 21202829
//Antoine Lecomte 21103457

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include "Chaine.h"
#include "Reseau.h"
#include "ArbreQuat.h"
#include "Hachage.h"

/**
 * @brief Génère une chaîne de points aléatoires.
 * @param numero Numéro de la chaîne.
 * @param nbPoints Nombre de points dans la chaîne.
 * @param xmax Coordonnée maximale en x.
 * @param ymax Coordonnée maximale en y.
 * @return CellChaine* Retourne la chaîne de points générée.
 */
CellChaine* genererChainePoints(int numero, int nbPoints, int xmax, int ymax){
    CellChaine* chaine = (CellChaine*)malloc(sizeof(CellChaine));
    if (!chaine){
        fprintf(stderr, "Allocation de mémoire pour la chaîne a échoué");
        exit(1);
    }
    chaine->numero = numero;
    chaine->points = NULL;
    chaine->suiv = NULL;
    for (int i = 0; i < nbPoints; i++){
        CellPoint* nouveau_point = (CellPoint*)malloc(sizeof(CellPoint));
        if (!nouveau_point){
            fprintf(stderr, "Allocation de mémoire pour le point a échoué");
            exit(1);
        }
        nouveau_point->x = (double)rand() / RAND_MAX * xmax;
        nouveau_point->y = (double)rand() / RAND_MAX * ymax;
        nouveau_point->suiv = chaine->points;
        chaine->points = nouveau_point;
    }
    return chaine;
}

/**
 * @brief Génère aléatoirement des chaînes de points.
 * @param nbChaines Nombre de chaînes à générer.
 * @param nbPointsChaine Nombre de points par chaîne.
 * @param xmax Coordonnée maximale en x.
 * @param ymax Coordonnée maximale en y.
 * @return Chaines* Retourne l'ensemble de chaînes de points générées.
 */
Chaines* generationAleatoire(int nbChaines, int nbPointsChaine, int xmax, int ymax){
    Chaines* ensembleChaines = (Chaines*)malloc(sizeof(Chaines));
    if (!ensembleChaines){
        fprintf(stderr, "Allocation de mémoire pour l'ensemble des chaînes a échoué");
        exit(1);
    }
    ensembleChaines->gamma = rand() % (INT_MAX - 2) + 3;
    ensembleChaines->nbChaines = nbChaines;
    ensembleChaines->chaines = NULL;
    for (int i = 0; i < nbChaines; i++){
        CellChaine* nouvelle_chaine = genererChainePoints(i + 1, nbPointsChaine, xmax, ymax);
        nouvelle_chaine->suiv = ensembleChaines->chaines;
        ensembleChaines->chaines = nouvelle_chaine;
    }
    return ensembleChaines;
}

/**
 * @brief Mesure le temps d'exécution d'une fonction.
 * @param func Fonction à mesurer.
 * @param C Chaînes de points.
 * @return double Temps d'exécution en secondes.
 */
double mesurer_temps_execution_chaine(Reseau* (*func)(Chaines*), Chaines* C){
    clock_t debut, fin;
    double temps_exec;
    debut = clock();
    Reseau* R = func(C);
    fin = clock();
    temps_exec = ((double) (fin - debut)) / CLOCKS_PER_SEC;
    libererReseau(R);
    return temps_exec;
}

/**
 * @brief Mesure le temps d'exécution d'une fonction utilisant la table de hachage.
 * @param func Fonction à mesurer.
 * @param C Chaînes de points.
 * @param M Taille de la table de hachage.
 * @return double Temps d'exécution en secondes.
 */
double mesurer_temps_execution_hachage(Reseau* (*func)(Chaines*, int), Chaines* C, int M){
    clock_t debut, fin;
    double temps_exec;
    debut = clock();
    Reseau* R = func(C,M);
    fin = clock();
    temps_exec = ((double) (fin - debut)) / CLOCKS_PER_SEC;
    libererReseau(R);
    return temps_exec;
}

/**
 * @brief Mesure le temps d'exécution d'une fonction utilisant l'arbre quaternaire.
 * @param func Fonction à mesurer.
 * @param C Chaînes de points.
 * @return double Temps d'exécution en secondes.
 */
double mesurer_temps_execution_arbrequat(Reseau* (*func)(Chaines*), Chaines* C){
    clock_t debut, fin;
    double temps_exec;
    debut = clock();
    Reseau* R = func(C);
    fin = clock();
    temps_exec = ((double) (fin - debut)) / CLOCKS_PER_SEC;
    libererReseau(R);
    return temps_exec;
}

/**
 * @brief Clone une liste de chaînes de points.
 * @param C Liste de chaînes à cloner.
 * @return Chaines* Le clone de la liste de chaînes.
 */
Chaines* cloneChaines(Chaines* C){
    Chaines* clone = (Chaines*)malloc(sizeof(Chaines));
    if (!clone){
        fprintf(stderr, "Allocation de mémoire pour le clone des chaînes a échoué\n");
        exit(1);
    }
    clone->gamma = C->gamma;
    clone->nbChaines = C->nbChaines;
    clone->chaines = NULL;
    CellChaine* current = C->chaines;
    CellChaine* prev = NULL;
    while (current != NULL){
        CellChaine* new_chaine = (CellChaine*)malloc(sizeof(CellChaine));
        if (!new_chaine){
            fprintf(stderr, "Allocation de mémoire pour une nouvelle chaîne a échoué\n");
            libererChaines(clone);
            exit(1);
        }
        new_chaine->numero = current->numero;
        new_chaine->points = NULL;
        new_chaine->suiv = NULL;

        if (clone->chaines == NULL){
            clone->chaines = new_chaine;
        }else{
            prev->suiv = new_chaine;
        }
        prev = new_chaine;
        CellPoint* current_point = current->points;
        CellPoint* prev_point = NULL;
        while (current_point != NULL){
            CellPoint* new_point = (CellPoint*)malloc(sizeof(CellPoint));
            if (!new_point){
                fprintf(stderr, "Allocation de mémoire pour un nouveau point a échoué\n");
                libererChaines(clone);
                exit(1);
            }
            new_point->x = current_point->x;
            new_point->y = current_point->y;
            new_point->suiv = NULL;

            if (new_chaine->points == NULL){
                new_chaine->points = new_point;
            }else{
                prev_point->suiv = new_point;
            }
            prev_point = new_point;
            current_point = current_point->suiv;
        }
        current = current->suiv;
    }
    return clone;
}


int main(int argc, char *argv[]){
    if (argc != 3){
        fprintf(stderr, "Usage: %s <fichier.cha> <nombre_iterations>\n", argv[0]);
        return 1;
    }

    Chaines *C;
    FILE *f;
    int nb_iterations = atoi(argv[2]);
    int aleatoire = strcmp(argv[1], "aleatoire") == 0;

    if (nb_iterations <= 0){
        fprintf(stderr, "Le nombre d'itérations doit être un entier positif.\n");
        return 1;
    }

    if (aleatoire){
        f = fopen("temps_exec_alea.txt", "w");
        if (!f){
            fprintf(stderr, "Erreur lors de l'ouverture du fichier pour écrire les données des points aléatoires\n");
            return 1;
        }
    }else{
        FILE *fichier_cha = fopen(argv[1], "r");
        if (!fichier_cha){
            fprintf(stderr, "Erreur lors de l'ouverture du fichier\n");
            return 1;
        }
        f = fopen("temps_exec.txt", "w");
        if (!f){
            fprintf(stderr, "Erreur lors de l'ouverture du fichier pour écrire les résultats\n");
            fclose(fichier_cha);
            return 1;
        }
        C = lectureChaines(fichier_cha);
        fclose(fichier_cha);
        if (!C){
            fprintf(stderr, "Erreur lors de la lecture du fichier %s\n", argv[1]);
            fclose(f);
            return 1;
        }
    }

    size_t tailles_table[] = {1, 10, 100, 1000};

    for (int nbChaines = 500; nbChaines <= 5000; nbChaines += 500){
        if (aleatoire){
            C = generationAleatoire(nbChaines, 100, 5000, 5000);
            fprintf(f, "Résultats pour %d chaînes :\n", nbChaines);
        }else{
            fprintf(f, "Résultats pour %d chaînes :\n", C->nbChaines);
        }
        
        double temps_total_liste = 0.0;
        double temps_total_arbre = 0.0;
        double temps_total_hachage[sizeof(tailles_table) / sizeof(tailles_table[0])] = {0.0};

        for (int j = 0; j < nb_iterations; j++){
            Chaines* chaine_nouv_1 = cloneChaines(C);
            temps_total_liste += mesurer_temps_execution_chaine(reconstitueReseauListe, chaine_nouv_1);
        }
        double temps_liste = temps_total_liste / nb_iterations;
        fprintf(f, "Temps moyen pour liste chaînée sur %d itérations: %.15f secondes\n", nb_iterations, temps_liste);
        fprintf(f, "Résultats pour table de hachage :\n");

        for (size_t i = 0; i < sizeof(tailles_table) / sizeof(tailles_table[0]); i++){
            int taille_table = tailles_table[i];
            double temps_hachage = 0.0;
            for (int j = 0; j < nb_iterations; j++){
                Chaines* chaine_nouv_2 = cloneChaines(C);
                temps_hachage += mesurer_temps_execution_hachage(reconstitueReseauHachage, chaine_nouv_2, taille_table);
            }
            temps_total_hachage[i] = temps_hachage;
            double temps_moyen_hachage = temps_hachage / nb_iterations;
            fprintf(f, "Temps moyen pour table de hachage (taille %d) sur %d itérations: %.15f secondes\n", taille_table, nb_iterations, temps_moyen_hachage);
        }
        fprintf(f, "Résultats pour arbre quaternaire :\n");
        double temps_arbre = 0.0;
        for (int j = 0; j < nb_iterations; j++){
            Chaines* chaine_nouv_3 = cloneChaines(C);
            temps_arbre += mesurer_temps_execution_arbrequat(reconstitueReseauArbre, chaine_nouv_3);
        }
        temps_total_arbre = temps_arbre;
        double temps_moyen_arbre = temps_arbre / nb_iterations;
        fprintf(f, "Temps moyen pour arbre quaternaire sur %d itérations: %.15f secondes\n\n", nb_iterations, temps_moyen_arbre);
        fprintf(f, "Temps totaux pour chaque méthode :\n");
        fprintf(f, "Temps total pour liste chaînée sur %d itérations: %.15f secondes\n", nb_iterations, temps_total_liste);

        for (size_t i = 0; i < sizeof(tailles_table) / sizeof(tailles_table[0]); i++){
            int taille_table = tailles_table[i];
            fprintf(f, "Temps total pour table de hachage (taille %d) sur %d itérations: %.15f secondes\n", taille_table, nb_iterations, temps_total_hachage[i]);
        }
        fprintf(f, "Temps total pour arbre quaternaire sur %d itérations: %.15f secondes\n", nb_iterations, temps_total_arbre);
        if (!aleatoire)
            break;
    }

    fclose(f);
    if (!aleatoire){
        libererChaines(C);
    }
    return 0;
}

