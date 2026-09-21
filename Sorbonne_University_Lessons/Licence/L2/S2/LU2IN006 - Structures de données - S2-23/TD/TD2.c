typedef struct element{
    int val;
    struct element *suiv;
    struct element *prec;
}Element;
typedef struct {
    Element *premier;
    Element *dernier;
}LDC;

/*Q1.4*/
void insererEnTete(LDC *ldc, int val){
    Element* e = creerElement(val);
    e->suiv = ldc->premier;
    if(listeVide(ldc)){
        ldc->dernier = e;
    }else{
        ldc->premier->prec = e;
    }
    ldc->premier = e;
}
/*Q1.5*/
void insererEnFin(LDC* ldc, int val){
    Element* e = creerElement(e);
    e->prec = ldc->dernier;
    if(listeVide(ldc)){
        ldc->premier = e;
    }else{
        ldc->dernier->suiv = e;
    }
    ldc->dernier = e;
}
/*Q1.6*/
void afficher(LDC* ldc){
    Element *e = ldc->dernier;
    while(e!=NULL){
        printf("%d", e->val);
        e = e->prec;
    }
    printf("\n");
}
/*Q1.7*/
Element *rechercher(LDC *ldc, int val){
    Element *e = ldc->premier;
    while(e!=NULL){
        if(e->val == val){
            return e;
        }
        e = e->suiv;
    }
    return NULL;
}
/*Q1.8*/
void supprimerElement(LDC* ldc, Element *e){
    Element *precE = e->prec;
    Element *suivE = e->suiv;
    if(precE==NULL){
        ldc->premier = suivE;
    }else{
        precE->suiv = suivE;
    }
    if(suivE==NULL){
        ldc->dernier = precE;
    }else{
        suivE->prec = precE;
    }
    free(e);
}
/*Q1.9*/
int supprimerTete(LDC *ldc){
    int val = ldc->premier->val;
    supprimerElement(ldc, val);
    return val;
}
/*Q1.10*/
void desalloueListe(LDC* ldc){
    while(ldc->premier != NULL){
        supprimerTete(ldc);
    }
    free(ldc);
}

//=======EX2=====//
