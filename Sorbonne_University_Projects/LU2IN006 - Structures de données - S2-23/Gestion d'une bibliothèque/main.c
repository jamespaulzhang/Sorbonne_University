#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include "entreeSortieLC.h"
#include "entreeSortieH.h"


void menu(){
    printf("Menu :\n");
    printf("0 - Sortie du programme\n");
    printf("1 - Affichage de la bibliothèque (liste chainée)\n");
    printf("2 - Insérer un ouvrage (liste chainée)\n");
    printf("3 - Recherche par numéro (liste chainée)\n");
    printf("4 - Recherche par titre (liste chainée)\n");
    printf("5 - Recherche par auteur (liste chainée)\n");
    printf("6 - Supprimer un ouvrage (liste chainée)\n");
    printf("7 - Fusionner avec une autre bibliothèque (liste chainée)\n");
    printf("8 - Rechercher les ouvrages avec plusieurs exemplaires (liste chainée)\n");
    printf("9 - Affichage de la bibliothèque (tableau de hachage)\n");
    printf("10 - Insérer un ouvrage (tableau de hachage)\n");
    printf("11 - Recherche par numéro (tableau de hachage)\n");
    printf("12 - Recherche par titre (tableau de hachage)\n");
    printf("13 - Recherche par auteur (tableau de hachage)\n");
    printf("14 - Supprimer un ouvrage (tableau de hachage)\n");
    printf("15 - Fusionner avec une autre bibliothèque (tableau de hachage)\n");
    printf("16 - Rechercher les ouvrages avec plusieurs exemplaires (tableau de hachage)\n");
    printf("Entrez le numéro de l'action souhaitée : ");
}

void calcul_temps(clock_t debut, clock_t fin){
    double temps = ((double)(fin-debut))/CLOCKS_PER_SEC;
    printf("Temps de recherche : %f secondes\n", temps);
}

/* EXERCICE 3.3
// Fonction pour mesurer le temps de recherche des ouvrages en plusieurs exemplaires
void mesurer_temps_recherche_multiples(Biblio* biblio, BiblioH* biblioH, int taille_biblio) {
    clock_t debut, fin;

    // Mesurer le temps de recherche des ouvrages en plusieurs exemplaires pour la liste chaînée
    debut = clock();
    Biblio* resultatLC = rechercher_exemplaires_multiples(biblio);
    fin = clock();
    double temps_recherche_LC = ((double)(fin - debut)) / CLOCKS_PER_SEC;

    // Mesurer le temps de recherche des ouvrages en plusieurs exemplaires pour le tableau de hachage
    debut = clock();
    BiblioH* resultatH = rechercher_exemplaires_multiplesH(biblioH);
    fin = clock();
    double temps_recherche_H = ((double)(fin - debut)) / CLOCKS_PER_SEC;

    // Afficher les temps de recherche pour cette taille de bibliothèque
    printf("Taille de la bibliothèque : %d\n", taille_biblio);
    printf("Temps de recherche (Liste chaînée) : %f secondes\n", temps_recherche_LC);
    printf("Temps de recherche (Tableau de hachage) : %f secondes\n", temps_recherche_H);

    // Libérer la mémoire des résultats
    if (resultatLC != NULL)
        liberer_biblio(resultatLC);
    if (resultatH != NULL)
        liberer_biblioH(resultatH);
}
*/

int main(){
    Biblio* biblio = creer_biblio();
    BiblioH* biblioH = creer_biblioH(100050); //Modifier la taille à 50000 dans l'exercice 3.3
    FILE* file = fopen("GdeBiblio.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier.\n");
        return 1;
    }

    int num;
    char titre[256];
    char auteur[256];
    char input[256];

    while (fscanf(file, "%d %s %s", &num, titre, auteur) == 3){
        inserer_en_tete(biblio, num, titre, auteur);
        inserer(biblioH, num, titre, auteur);
    }

    fclose(file);

/* EXERCICE 3.3
    int taille_biblio = 1000;
    while (taille_biblio <= 50000) {
        mesurer_temps_recherche_multiples(biblio, biblioH, taille_biblio);
        taille_biblio *= 2; // Multiplier par 2 à chaque itération
    }
*/


    clock_t debut, fin;

    int rep;
    do{
        menu();
        fgets(input, sizeof(input), stdin); 
        if (sscanf(input, "%d", &rep) != 1) {
            printf("Entrée non valide. Veuillez saisir un numéro valide.\n");
            while (getchar() != '\n');
            continue;
        }
        switch (rep){
            case 0:
                printf("Sortie du programme.\n");
                break;
            case 1:
                printf("Affichage :\n");
                afficher_biblio(biblio);
                break;
            case 2: {
                printf("Insertion d'un nouvel ouvrage (liste chainée):\n");
                do {
                    printf("Entrez le numéro (numéro doit être supérieur ou égale à 0, ou entrez 'q' pour quitter) : ");
                    fgets(input, sizeof(input), stdin);
                    if (input[0] == 'q' || input[0] == 'Q'){
                        printf("Quitter l'insertion.\n");
                        break;
                    }
                    sscanf(input, "%d", &num);
                    if (num < 0){
                        printf("Numéro invalide. Le numéro doit être supérieur ou égale à 0.\n");
                        continue;
                    }
                    if (rechercher_par_numero(biblio, num) != NULL){
                        printf("Le numéro existe déjà dans la bibliothèque. Veuillez saisir un autre numéro.\n");
                        continue;
                    }
                    break;
                }while (1);

                if (input[0] == 'q' || input[0] == 'Q'){
                    break;
                }

                printf("Entrez le titre et l'auteur de l'ouvrage : ");
                fgets(input, sizeof(input), stdin);
                sscanf(input, "%s %s", titre, auteur);
                inserer_en_tete(biblio, num, titre, auteur);
                printf("Ajout fait.\n");
                break;
            }
            case 3:{
                printf("Recherche par numéro (liste chainée):\n");
                printf("Entrez le numéro de l'ouvrage : ");
                fgets(input, sizeof(input), stdin);
                sscanf(input, "%d", &num);

                debut = clock();
                Livre* livre = rechercher_par_numero(biblio, num);
                fin = clock();

                if (livre != NULL) {
                    printf("Ouvrage trouvé : ");
                    afficher_livre(livre);
                }
                else{
                    printf("Aucun ouvrage trouvé avec ce numéro.\n");
                }
                printf("numéro (LC) : ");
                calcul_temps(debut, fin);
                break;
            }
            case 4:{
                printf("Recherche par titre (liste chainée):\n");
                printf("Entrez le titre de l'ouvrage : ");
                fgets(input, sizeof(input), stdin);
                sscanf(input, "%s", titre);

                debut = clock();
                Livre* livre = rechercher_par_titre(biblio, titre);
                fin = clock();

                if (livre != NULL){
                    printf("Ouvrage trouvé : ");
                    afficher_livre(livre);
                }
                else{
                    printf("Aucun ouvrage trouvé avec ce titre.\n");
                }
                printf("titre (LC) : ");
                calcul_temps(debut, fin);
                break;
            }
            case 5:{
                printf("Recherche par l'auteur (liste chainée):\n");
                printf("Entrez l'auteur de l'ouvrage : ");
                fgets(input, sizeof(input), stdin);
                sscanf(input, "%s", auteur);

                debut = clock();
                Biblio* resultat = rechercher_par_auteur(biblio, auteur);
                fin = clock();

                if (resultat != NULL) {
                    printf("Ouvrages trouvés de l'auteur %s :\n", auteur);
                    afficher_biblio(resultat);
                }
                else{
                    printf("Aucun ouvrage trouvé de l'auteur %s.\n", auteur);
                }
                printf("auteur (LC) : ");
                calcul_temps(debut, fin);
                break;
            }
            case 6:{
                printf("Supprimer un ouvrage.(liste chainée)\n");
                do{
                    printf("Entrez le numéro, le titre et l'auteur de l'ouvrage à supprimer : ");
                    fgets(input, sizeof(input), stdin);
                }while(sscanf(input, "%d %s %s", &num, titre, auteur) != 3);
                supprimer_livre(biblio, num, titre, auteur);
                printf("Suppression faite.\n");
                break;
            }
            case 7:{
                printf("Fusionner avec une autre bibliothèque (liste chainée):\n");
                printf("Entrez le nom du fichier contenant la bibliothèque à fusionner : ");
                fgets(input, sizeof(input), stdin);
                sscanf(input, "%s", titre);
                Biblio* biblio2 = charger_n_entrees(titre, INT_MAX);
                fusionner_bibliotheques(biblio, biblio2);
                printf("Fusion terminée.\n");
                break;
            }
            case 8:{
                printf("Rechercher les ouvrages avec plusieurs exemplaires (liste chainée):\n");

                debut = clock();
                Biblio* resultat = rechercher_exemplaires_multiples(biblio);
                fin = clock();

                if (resultat != NULL) {
                    printf("Ouvrages avec plusieurs exemplaires trouvés :\n");
                    afficher_biblio(resultat);
                    liberer_biblio(resultat);
                }
                else{
                    printf("Aucun ouvrage avec plusieurs exemplaires trouvé.\n");
                }
                printf("plusieurs exemplaires (LC) : ");
                calcul_temps(debut, fin);
                break;
            }
            case 9:
                printf("Affichage de la bibliothèque (tableau de hachage):\n");
                afficher_biblioH(biblioH);
                break;
            case 10: {
                printf("Insérer un ouvrage (tableau de hachage):\n");
                do {
                    printf("Entrez le numéro (numéro doit être supérieur ou égal à 0, ou entrez 'q' pour quitter) : ");
                    fgets(input, sizeof(input), stdin);
                    if (input[0] == 'q' || input[0] == 'Q') {
                        printf("Quitter l'insertion.\n");
                        break;
                    }
                    sscanf(input, "%d", &num);
                    if (num < 0) {
                        printf("Numéro invalide. Le numéro doit être supérieur ou égal à 0.\n");
                        continue;
                    }
                    if (rechercher_par_numeroH(biblioH, num) != NULL) {
                        printf("Le numéro existe déjà dans la bibliothèque. Veuillez saisir un autre numéro.\n");
                        continue;
                    }
                    break;
                } while (1);

                if (input[0] == 'q' || input[0] == 'Q') {
                    break;
                }

                printf("Entrez le titre et l'auteur de l'ouvrage : ");
                fgets(input, sizeof(input), stdin);
                sscanf(input, "%s %s", titre, auteur);
                inserer(biblioH, num, titre, auteur); 
                
                // Vérifier si l'insertion a réussi en vérifiant si le numéro existe maintenant dans la bibliothèque
                if (rechercher_par_numeroH(biblioH, num) != NULL) {
                    printf("Ouvrage inséré.\n");
                } else {
                    printf("Échec de l'insertion de l'ouvrage.\n");
                }
                break;
            }
            case 11:{
                printf("Recherche par numéro (tableau de hachage):\n");
                printf("Entrez le numéro de l'ouvrage : ");
                fgets(input, sizeof(input), stdin);
                sscanf(input, "%d", &num);

                debut = clock();
                LivreH* livre = rechercher_par_numeroH(biblioH, num);
                fin = clock();

                if (livre != NULL) {
                    printf("Ouvrage trouvé : ");
                    afficher_livreH(livre);
                }
                else{
                    printf("Aucun ouvrage trouvé avec ce numéro.\n");
                }
                printf("numéro (H) : ");
                calcul_temps(debut, fin);
                break;
            }
            case 12:{
                printf("Recherche par titre (tableau de hachage):\n");
                printf("Entrez le titre de l'ouvrage : ");
                fgets(input, sizeof(input), stdin);
                sscanf(input, "%s", titre);

                debut = clock();
                LivreH* livre = rechercher_par_titreH(biblioH, titre);
                fin = clock();
                
                if (livre != NULL){
                    printf("Ouvrage trouvé : ");
                    afficher_livreH(livre);
                }
                else{
                    printf("Aucun ouvrage trouvé avec ce titre.\n");
                }
                printf("titre (H) : ");
                calcul_temps(debut, fin);
                break;
            }
            case 13:{
                printf("Recherche par l'auteur (tableau de hachage):\n");
                printf("Entrez l'auteur de l'ouvrage : ");
                fgets(input, sizeof(input), stdin);
                sscanf(input, "%s", auteur);

                debut = clock();
                BiblioH* resultat = rechercher_par_auteurH(biblioH, auteur);
                fin = clock();

                if (resultat != NULL) {
                    printf("Ouvrages trouvés de l'auteur %s :\n", auteur);
                    afficher_biblioH(resultat);
                }
                else{
                    printf("Aucun ouvrage trouvé de l'auteur %s.\n", auteur);
                }
                printf("auteur (H) : ");
                calcul_temps(debut, fin);
                break;
            }
            case 14:{
                printf("Supprimer un ouvrage (tableau de hachage):\n");
                printf("Entrez le numéro, le titre et l'auteur de l'ouvrage à supprimer : ");
                fgets(input, sizeof(input), stdin);
                sscanf(input, "%d %s %s", &num, titre, auteur);
                supprimer_livreH(biblioH, num, titre, auteur);
                break;
            }
            case 15:{
                printf("Fusionner avec une autre bibliothèque (tableau de hachage):\n");
                printf("Entrez le nom du fichier contenant la bibliothèque à fusionner : ");
                fgets(input, sizeof(input), stdin);
                sscanf(input, "%s", titre);
                BiblioH* biblio2H = charger_n_entreesH(titre, INT_MAX);
                fusionner_bibliothequesH(biblioH, biblio2H);
                printf("Bibliothèques fusionnées.\n");
                break;
            }
            case 16:{
                printf("Rechercher les ouvrages avec plusieurs exemplaires (tableau de hachage):\n");

                debut = clock();
                BiblioH* resultat = rechercher_exemplaires_multiplesH(biblioH);
                fin = clock();

                if (resultat != NULL) {
                    printf("Ouvrages avec plusieurs exemplaires trouvés :\n");
                    afficher_biblioH(resultat);
                    liberer_biblioH(resultat);
                }
                else{
                    printf("Aucun ouvrage avec plusieurs exemplaires trouvé.\n");
                }
                printf("plusieurs exemplaires (H) : ");
                calcul_temps(debut, fin);
                break;
            }
            default:
                printf("Option invalide. Veuillez choisir une option valide.\n");
        }
    }while (rep != 0);

    printf("Merci et au revoir.\n");

    liberer_biblio(biblio);
    liberer_biblioH(biblioH);

    return 0;
}

//valgrind --leak-check=full ./main
