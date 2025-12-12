//===========EX1========//
/*Q1.1*/
Noeud* rechercherValeur(ABR ab, int val){
    if(ab == NULL)
        return NULL;
    if(ab->valeur == val)
        return ab;
    if(ab->valeur > val)
        return rechercherValeur(ab->fg, val);
    return rechercherValeur(ab->fd, val);
}
/*Q1.2*/
void insererElem(ABR* ab, int val){
    if(*ab == NULL){
        Noeud* n = (Noeud*)malloc(sizeof(Noeud));
        n->valeur = val;
        n->fg = NULL;
        n->fd = NULL;
        *ab = n;
    }else{
        if((*ab)->valeur > val){
            insererElem(&((*ab)->fg), val);
        }else{
            insererElem(&((*ab)->fd), val);
        }
    }
}
void supprimer(ABR* ab, int val){
    if(*ab == NULL)
        return;
    if((*ab)->valeur > val){
        supprimer(&((*ab)->fg), val);
        return;
    }
    if((*ab)->valeur < val){
        supprimer(&((*ab)->fd), val);
    }
    if((*ab)->fg == NULL){
        Noeud* n = (*ab)->fd;
        free(*ab);
        *ab = n;
        return;
    }
    if((*ab)->fd == NULL){
        Noeud* n = (*ab)->fg;
        free(*ab);
        *ab = n;
        return;
    }
    Noeud** cour = &((*ab)->fg);
    while((*cour)->fd != NULL){
        cour = &((*cour)->fd);
    }
    (*ab)->val = (*cour)->val;
    supprimer(cour,(*cour)->val);
}

//========EX2=======//
/*Q2.3*/
int AB_hauter(ABR ab){
    if(ab == NULL)  return 0;
    return ab->hauteur;
}
void majHauter(ABR ab){
    if(ab != NULL)
        ab->hauteur = 1 + max(majHauter(ab->fg), majHauter(ab->fd));
}
int max(int a, int b){
    if(a>b) return a;
    return b;
}
/*Q2.4*/
void rotationDroite(ABR* ab){
    Noeud* r = *ab;
    Noeud* g = r->fg;
    Noeud* v = g->fd;
    *ab = g;
    g->fd = r;
    r->fg = v;
    majHauter((*ab)->fd);
    majHauter(*ab);
}
/*Q2.6*/
void insererElem_avec_eq(ABR* ab, int v){
    if(*ab == NULL){
        //...
    }
    if((*ab)->val > v){
        //...
    }
}