/* T-BOX */
subs(chat,felin).                                    /* chat ⊑ felin */ /* Les chats sont des felins */
subs(felin,mammifere).
subs(mammifere,animal).
subs(canide,mammifere).
subs(chien,canide).
subs(canide,chien).
subs(canari,animal).
subs(animal,etreVivant).
subs(lion,felin).
subs(lion,carnivoreExc).
subs(carnivoreExc,predateur).
subs(chihuahua,and(chien,pet)).                      /* chihuahua ⊑ chien ⊓ pet */ /* Un chihuahua est a la fois un chien et un animal de compagnie. */
subs(souris,mammifere).
subs(and(animal,some(aMaitre)),pet).                 /* animal ⊓ ∃aMaitre ⊑ pet */ /* Un animal qui a un maitre est un animal de compagnie */
subs(pet,some(aMaitre)).
subs(animal,some(mange)).
subs(some(aMaitre),all(aMaitre,personne)).           /* ∃aMaitre ⊑ ∀aMaitre.personne */ /* Toute entite qui a un maitre ne peut avoir quun (ou plusieurs) maitre(s) humain(s) */
subs(and(animal,plante),nothing).
subs(and(all(mange,nothing),some(mange)),nothing).   /* ∀mange.nothing ⊓ ∃mange ⊑ nothing */ /* On ne peut pas a la foisne rien manger (ne manger que des choses qui n’existent pas) et manger quelque chose. */
equiv(carnivoreExc,all(mange,animal)).               /* carnivoreExc ≡ ∀mange.animal */ /* Un carnivore exclusif est d´efini comme une entit´e qui mange uniquement des animaux */
equiv(herbivoreExc,all(mange,plante)).
