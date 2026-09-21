typedef struct element{
    int val;
    int def;
    struct element* suiv;
}Element;
typedef struct{
    int tailleMax;
    Element** T;
}TableHash;

/*Q4.1*/
TableHash* H = (TableHash*)malloc(sizeof(TableHash));
H->tailleMax = 16;
H->T = (Element**)malloc(16*sizeof(Element*));
for(int i = 0; i < H->tailleMax; i++){
    H->T[i] = NULL;
}
/*Q4.2*/
void insertion(TableHash* H, int v, int c){
    int pos = g(c);
    Element* new = (Element*)malloc(sizeof(Element));
    new->val = v;
    new->def = c;
    new->suiv = H->T[pos];
    H->T[pos] = new;
}