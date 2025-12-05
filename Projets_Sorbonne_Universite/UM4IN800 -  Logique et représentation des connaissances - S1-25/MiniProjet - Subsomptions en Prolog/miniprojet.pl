/* Binôme : [Yuxiang ZHANG, Kenan ALSAFAD] */

/* T-BOX - Toutes les définitions subs/2 ensemble */
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
subs(and(all(mange,nothing),some(mange)),nothing).   /* ∀mange.nothing ⊓ ∃mange ⊑ nothing */ /* On ne peut pas a la foisne rien manger (ne manger que des choses qui n'existent pas) et manger quelque chose. */

% Règles pour les équivalences (Question 2.6)
subs(C, D) :- equiv(C, D).
subs(D, C) :- equiv(C, D).

% Définitions des équivalences
equiv(carnivoreExc,all(mange,animal)).               /* carnivoreExc ≡ ∀mange.animal */ /* Un carnivore exclusif est défini comme une entité qui mange uniquement des animaux */
equiv(herbivoreExc,all(mange,plante)).

% Règles de subsomption initiales (Question 2.1)
subsS1(C,C).
subsS1(C,D) :- subs(C,D),C\==D.
subsS1(C,D) :- subs(C,E),subsS1(E,D).

% Règles de subsomption avec prévention des cycles (Question 2.3)
subsS(C,D) :- subsS(C,D,[C]).
subsS(C,C,_).
subsS(C,D,_):-subs(C,D),C\==D.
subsS(C,D,L):-subs(C,E),not(member(E,L)),subsS(E,D,[E|L]),E\==D.

% Règles pour gérer l'intersection
subsS(C,and(D1,D2),L) :- D1\=D2,subsS(C,D1,L),subsS(C,D2,L).
subsS(C,D,L) :- subs(and(D1,D2),D),E=and(D1,D2),not(member(E,L)), subsS(C,E,[E|L]),E\==C.
subsS(and(C,C),D,L) :- nonvar(C),subsS(C,D,[C|L]).
subsS(and(C1,C2),D,L) :- C1\=C2,subsS(C1,D,[C1|L]).
subsS(and(C1,C2),D,L) :- C1\=C2,subsS(C2,D,[C2|L]).
subsS(and(C1,C2),D,L) :- subs(C1,E1),E=and(E1,C2),not(member(E,L)),subsS(E,D,[E|L]),E\==D.
subsS(and(C1,C2),D,L) :- Cinv=and(C2,C1),not(member(Cinv,L)),subsS(Cinv,D,[Cinv|L]).

% Règles permettant de répondre à une requête du type ∀R.C ⊑s ∀R.D (Question 4.1)
subsS(all(R,C),all(R,D),L) :- subsS(C,D,L).

% Règle pour les concepts all vers les concepts atomiques équivalents (Question 4.2)
subsS(all(R,C),Atomic,L) :- 
    equiv(Atomic,all(R,D)), 
    subsS(C,D,L).

% Règle pour les concepts atomiques équivalents vers les concepts all (Question 4.2)
subsS(Atomic,all(R,C),L) :-
    equiv(Atomic,all(R,D)),
    subsS(D,C,L).

% Règle pour la conjonction de restrictions universelles (Question 4.3)
subsS(and(all(R,C),all(R,D)),all(R,E),L) :- 
    subsS(and(C,D),E,L).