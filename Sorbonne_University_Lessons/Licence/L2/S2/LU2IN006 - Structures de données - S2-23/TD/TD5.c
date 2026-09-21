//=======EX2=======//
/*Q2.4*/
typedef struct tas{
    int *tab;
}TasTernaire;

int supprimerRacine(TasTernaire* tas){
    int racine = tas->tab[1];
    int nb = tas->tab[0];
    int feuille = tas->tab[nb];
    tas->tab[0] -= 1;
    int pos = 1;
    while(3*pos-1 < nb){
        int min = tas->tab[3*pos-1];
        int fils = -1;
        if(3*pos <= nb && tas->tab[3*pos] < min){
            min = tas->tab[3*pos];
            fils = 0;
        }
        if(3*pos+1 <= nb && tas->tab[3*pos+1] < min){
            min = tas->tab[3*pos+1];
            fils = 1;
        }
        if(min < feuille){
            tas->tab[pos] = min;
            pos = 3*pos+fils;
        }else{
            tas->tab[pos] = feuille;
            return racine;
        }
    }tas->tab[pos] = feuille;
    return racine;
}