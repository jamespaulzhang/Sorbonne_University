#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int somme_carres_differences(int tableau[], int n) {
    int somme = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int difference = tableau[i] - tableau[j];
            somme += difference * difference;
        }
    }
    return somme;
}

int somme_carres_differences_2(int tableau[], int n) {
	int res = 0, s1 = 0, s2 = 0;
	for (int i = 0; i < n; i++) {
		s1 += (tableau[i])*(tableau[i]);
		s2 += tableau[i];
	}
	s2 *= s2;
	res = 2*n*s1-2*s2;
	return res;
}

int main() {
    FILE* f = fopen("sortie_vitesse.txt", "w");

    for (int n = 1; n <= 100; n++) {
        int* tableau = malloc(sizeof(int) * n);
        // Initialiser le tableau avec des valeurs, si nécessaire

        clock_t temps_initial, temps_final;
        double temps_cpu;

        temps_initial = clock();
        somme_carres_differences(tableau, n);
        temps_final = clock();

        temps_cpu = ((double)(temps_final - temps_initial)) / CLOCKS_PER_SEC;

        fprintf(f, "%d\t%f\t", n, temps_cpu);

        temps_initial = clock();
        somme_carres_differences_2(tableau, n);
        temps_final = clock();

        temps_cpu = ((double)(temps_final - temps_initial)) / CLOCKS_PER_SEC;

        fprintf(f, "%f\n", temps_cpu);

        free(tableau);
    }

    fclose(f);
    return 0;
}
