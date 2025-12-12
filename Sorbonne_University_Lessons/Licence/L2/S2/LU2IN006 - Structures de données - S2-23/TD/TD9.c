//==========EX1=========//
/*Q1.3*/
typedef struct cell{
    int val;
    struct cell* next;
    struct cell* below;
}Cell;
typedef struct layer{
    Cell* first;
    struct layer* below;
    struct layer* above;
    int level;
}Layer;
typedef struct{
    Layer* top;
    Layer* bottom;
    int nbLevel;
    float p;
}Skiplist;
/*Q1.4*/
void skiplist_print(Skiplist* sl){
    Layer* cur_layer = sl->top;
    while(cur_layer != NULL){
        printf("%d:", cur_layer->level);
        Cell* cur_cell = cur_layer->first;
        while(cur_cell != NULL){
            printf("->%d", cur_cell->val);
            cur_cell = cur_cell->next;
        }
        printf("\n");
        cur_layer = cur_layer->below;
    }
}
/*Q1.6*/
Cell** path(Skiplist* sl, int val){
    Cell** path = (Cell**)malloc((*sl)->nbLevel*sizeof(Cell*));
    Layer* layer = sl->top;
    while(layer != NULL && layer->first->val > val){
        path[layer->level] = NULL;
        layer = layer->below;
    }
    if(layer == NULL)   return path;
    Cell* c = layer->first;
    while(c != NULL){
        if(c->next == NULL || c->next->val > val){
            path[layer->level] = c;
            c = c->below;
            layer = layer->below;
        }else{
            c = c->next;
        }
    }
    return path;
}
void add_layer(Skiplist* sl, Cell* c){
    Layer* new = (Layer*)malloc(sizeof(Layer));
    new->first = c;
    new->above = NULL;
    new->below = sl->top;
    new->level = 1+sl->nbLevel;
    if(sl->top != NULL){
        sl->top->above = new;
    }
    if(sl->bottom == NULL){
        sl->bottom = new;
    }
    sl->top = new;
}
void insert_rec(Skiplist* sl, int val, Cell** path, Layer* layer, Cell* below){
    Cell* new = (Cell*)malloc(sizeof(Cell));
    new->val = val;
    new->below = below;
    if(layer == NULL){
        add_layer(sl, new);
        if(isRandomOK(sl)){
            //......
        }
    }
}
