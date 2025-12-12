//==========EX1=======//
/*Q1.3*/
typedef struct arete{
    int u,v;
    double dist;
}Arete;
typedef ElementListeA* ListeA;
typedef struct sommet{
    float x,y;
    char* nomVille;
    ListeA L_adj;
    int numS;
}Sommet;
typedef struct elementListeA{
    Arete *a;
    struct elementListeA* suiv;
}ElementListeA;
typedef struct graphe{
    int nbsom;
    Sommet** tabS;
}Graphe;
/*Q1.5*/
Graphe* creerGraphe(int n){
    Graphe* g = (Graphe*)malloc(sizeof(Graphe));
    g->nbsom = n;
    g->tabS = (Sommet**)malloc(n*sizeof(Sommet*));
    for(int i = 0; i < n; i++){
        g->tabS[i] = (Sommet*)malloc(sizeof(Sommet));
        g->tabS[i]->numS = i;
        g->tabS[i]->L_adj = NULL;
    }
    return g;
}
/*Q1.6*/
void majSommet(Graphe* g, int i, char* nom, float x, float y){
    g->tabS[i]->nomVille = strdup(nom);
    g->tabS[i]->x = x;
    g->tabS[i]->y = y;
}
/*Q1.8*/
Arete* creerArete(int u, int v, double dist){
    Arete* a = (Arete*)malloc(siweof(Arete));
    a->u = u;
    a->v = v;
    a->dist = dist;
    return a;
}
void insererTeteListeA(ListeA *l, Arete* a){
    ElementListeA *new = (ElementListeA*)malloc(sizeof(ElementListeA));
    new->a = a;
    new->suiv = *l;
    *l = new;
}
void ajouteArete(Graphe* g, int i, int j, double dist){
    Arete* a = creerArete(i, j, dist);
    insererTeteListeA(&(g->tab[i]->L_adj), a);
    insererTeteListeA(&(g->tab[j]->L_adj), a);
}
/*Q1.9*/
void afficheListA(ElementListeA *l){
    ElementListeA* p = *l;
    while(p != NULL){
        printf("...");
        p = p->suiv;
    }
}
void afficheGraphe(Graphe* g){
    for(int i = 0; i < g->nbsom;i++){
        printf("...");
        afficheListA(g->tabS[i]->L_adj);
    }
}
/*Q1.10*/
void desalloueListeA(ListeA l){
    while(l != NULL){
        ElementListeA* suiv = l->suiv;
        if(l->a->u == -1){
            free(l->a);
        }else{
            //.....
        }
    }
}
