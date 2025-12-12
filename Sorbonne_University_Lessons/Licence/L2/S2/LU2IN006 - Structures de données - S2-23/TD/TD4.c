/*Q1.1*/
typedef struct Form Formation;
typedef struct element{
    Formation* f;
    struct element* suiv;
}Element;
struct Form{
    char* nom;
    int nbheurs;
    Element* L_form;
}Formation;
/*Q1.2*/
typedef struct{
    int M;
    int num;
    Formation* T;
}Catalogue;
/*Q1.4*/
void afficher_formation(Formation* f){
    if(f->nbheurs!=0){
        printf("Cours %s\n", f->nom);
    }else{
        printf("Formation %s", f->nom);
        Element* cour = f->L_form;
        while(cour != NULL){
            printf("%s", cour->f->nom);
            cour = cour->suiv;
        }
        printf("\n");
    }
}
void affichage_catalogue(Catalogue* c){
    for(int i = 0; i < c->num; i++){
        afficher_formation((c->T)[i]);
    }
}
/*Q1.5*/
int nb_heures_total(Formation *f){
    int heures = f->nbheurs;
    Element* cour = f->L_form;
    while(cour!= NULL){
        heures += nb_heures_total(cour->f);
        cour = cour->suiv;
    }
    return heures;
}

//======EX2======//
/*Q2.1*/
typedef struct maison{
    char* ssoc;
    struct maison *suiv;
}Maison;
typedef struct assoc{
    char* nomAssoc;
    Element* lesMembres;
}Association;
typedef struct element{
    char* nom;
    struct element *suivant;
}Element;
/*Q2.2*/
Association *creerAssociation(char *nom){
    Association* a = (Association*)malloc(sizeof(Association));
    a->nomAssoc = strdup(nom);
    a->lesMembres = NULL;
    return a;
}
void ajouterPersonne(Association *a, char *nom){
    Element* membre = (Element*)malloc(sizeof(Element));
    membre->nom = strdup(nom);
    member->suivant = a->lesMembres;
    a->lesMembres = member;
}
void supprimerAssociation(Association* a, char* nom){
    Element* cours = a->lesMembres;
    Element* prec = NULL;
    while(cours != NULL && strcmp(cours->nom, nom)!=0){
        prec = cours;
        cours = cours->suiv;
    }
}

typedef struct assoc{
    char* nom;
    char* addresse;
    Membre *listeMembre;
}Assocation;
typedef struct membre{
    Association *addresse;
    struct membre* suiv;
}Membre;
