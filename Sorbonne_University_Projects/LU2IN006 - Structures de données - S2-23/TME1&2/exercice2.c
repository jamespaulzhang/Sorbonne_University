#include<stdio.h>
#include<time.h>

int* alloue_tableau_1(int n){
    int* tab = (int*)malloc(sizeof(int)*n);
    return tab;
}

void alloue_tableau_2(int **T,int n){
    *T = (int*)malloc(n*sizeof(n));
}

void desalloue_tableau(int *tab){
    free(tab);
}

void remplirTableau(int *tab, int n, int V) {
    srand(time(NULL));

    for (int i = 0; i < n; i++) {
        tab[i] = rand() % V;
    }
}

void afficher_tableau(int* tab,int taille){
    for(int i = 0;i < taille ; i++){
        printf("%d\n",tab[i]);
    }
}