Yuxiang Zhang et Antoine Lecomte


TME 5 - ECC


Question 3 :

Le symbole de Legendre (a/p) est utilisé pour trouver si un entier a est un résidu quadratique modulo p, donc pour déterminer s'il existe ou non un entier x tel que : 
x² ≡ a(modp), où p est un nombre premier impair.
L’expression a^((p−1)/2) mod p vient du petit théorème de Fermat et du critère d'Euler :
soit p un nombre premier impair et a un entier non divisible par p, alors :
(a/p) ≡ a^((p−1)/2) (mod p) car :
- si a est un résidu quadratique mod p, alors a^((p-1)/2) ≡ 1 mod p.
- si a n'est pas un résidu quadratique mod p, alors a^((p-1)/2) ≡ -1 mod p.
- si a est un diviseur de p, alors a^((p-1)/2) ≡ 0 mod p.
Donc la formule correspond parfaitement à la définition du symbole de Legendre.


Question 5 :

Soit x = a^((p+1)/4) mod p. Montrons que x² ≡ a mod p.
x² = (a^((p+1)/4))² = a^((p+1)/2). On multiplie a^((p+1)/2) = a*a^((p-1)/2) ≡ a*1 = a mod p, ce qui donne x² ≡ a mod p. (x est donc une racine carrée de a mod p)


Question 6 :

Enoncé du théorème de Hasse :
Soit E une courbe elliptique définie sur le corps fini Fp, donnée par une équation de la forme :
E : y² = x^3 + ax + b, avec a,b ∈ Fp et telle que le discriminant Δ = −16(4a^3 + 27b²) soit non nul modulo p. Le nombre de points N de la courbe E sur Fp, point à l'infini compris, vérifie : ∣N−(p+1)∣ ≤ 2*sqrt(p).


Question 11 :

Pour estimer combien de tirages aléatoires sont nécessaires avant de tomber sur un point (x,y) qui appartient à une courbe elliptique E définie sur un corps fini Fp :
La courbe elliptique est donnée par une équation de la forme : E : y² = x^3 + ax + b mod p.
Il y a p valeurs possibles pour x, et p valeurs possibles pour y, donc p² couples (x,y) possibles au total. ∣N−(p+1)∣≤2*sqrt(p) ⇒ N≈p, d'après le théorème de Hasse pour le nombre de points N, et puisque l'ensemble total des couples (x,y) ∈ Fp² est de taille p² et que la courbe elliptique contient environ p points, la probabilité qu'un couple (x,y) aléatoire appartienne à E est : P((x,y) ∈ E) ≈ p/p² = 1/p. La valeur moyenne du nombre de tentatives avant de réussir un tirage est donnée par l'espérance :
E[nombre de tirages] = 1/P(succès) ≈ p. A chaque tentative on choisit x et y aléatoirement et on vérifie si (x,y) satisfait l'équation y² = x^3 + ax + b mod p. Cette vérification est en temps constant soit en O(1), mais comme il faut p tirages, la complexité globale est O(p).

Commentaire sur la fonction : L'algorithme tourne sans trouver de point sur la courbe rapidement, nous avons interrompu la recherche.


Question 12 :

Complexité : 
- Tirage aléatoire de x ∈ Fp en O(1).
- Calcul de rhs en O(1).
- Calcul du symbole de Legendre (rhs/p) en O(log p).
- Si rhs est un carré, calcul de la racine carrée en O(log p).
On suppose que pour environ la moitié des x, rhs est un résidu quadratique donc la probabilité de succès est ≈ 1/2, donc cela fait un nombre moyen d'itérations de 2 pour avoir un succès. On améliore donc la version point_aleatoire_naif qui a une complexité de O(p) en la fonction plus performante point_aleatoire ayant une complexité donc de O(log p).

Commentaire sur la fonction : Résultat très rapide, en quelques secondes on trouve un point de la courbe elliptique : (243259240467900607245171, 302840267118854060028797)

le script utilisé est le suivant :

from ecc import *

E = (360040014289779780338359, 117235701958358085919867, 18575864837248358617992)
print(point_aleatoire(E))

#(243259240467900607245171, 302840267118854060028797)


Question 15 :

Pour l'échange de clés Diffie-Hellman sur une courbe elliptique on a besoin d'un point P ∈ E qui engendre un grand sous-groupe cyclique de E (Fp) qui soit idéalement premier. On veut trouver P tel que son ordre soit q pour qu'il génère un sous-groupe d'ordre q.
C'est un bon point car le problème DLP dans un groupe d'ordre premier est difficile, il n'y a pas de petits sous-groupes qui peuvent diminuer la sécurité de l'échange de clés et puisque q est très grand le protocole est bien sécurisé.
Pour trouver P : tirer un point aléatoire R ∈ E avec point_aleatoire et calculer :
P = N/q*R = 4R. Vérifier ensuite que P != O (point à l’infini), que q*P = O, donc que l'ordre de P divise q et que donc si q*P = O et P != O, alors ordre(P) = q.
Nous avons trouvé P = (187457356424232318540988588897979445026, 44311955355832672245737191477667626255)

à l'aide du script suivant :

from ecc import *

p = 248301763022729027652019747568375012323
E = (p, 1, 0)
N = 248301763022729027652019747568375012324
factors_N = [(2, 2), (62075440755682256913004936892093753081, 1)]
n = 62075440755682256913004936892093753081  # facteur premier
P = point_ordre(E, N, factors_N, n)
assert ordre(N, factors_N, P, E) == n
print (P)