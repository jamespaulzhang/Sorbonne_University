Antoine Lecomte & Yuxiang Zhang


Question 9 :

Après lancement du test 5, l'affichage est le suivant :

----------------------------------------------


Test 5 : Cryptanalyse V1
---------------------
Test cryptanalyse_v1
Cryptanalysis of data/text1 = SUCCESS
Cryptanalysis of data/text2 = FAILED
Cryptanalysis of data/text3 = SUCCESS
Cryptanalysis of data/text4 = FAILED
Cryptanalysis of data/text5 = FAILED
Cryptanalysis of data/text6 = FAILED
Cryptanalysis of data/text7 = FAILED
Cryptanalysis of data/text8 = SUCCESS
Cryptanalysis of data/text9 = FAILED
Cryptanalysis of data/text10 = SUCCESS
Cryptanalysis of data/text11 = FAILED
Cryptanalysis of data/text12 = FAILED
Cryptanalysis of data/text13 = FAILED
Cryptanalysis of data/text14 = SUCCESS
Cryptanalysis of data/text15 = SUCCESS
Cryptanalysis of data/text16 = FAILED
Cryptanalysis of data/text17 = SUCCESS
Cryptanalysis of data/text18 = FAILED
Cryptanalysis of data/text19 = FAILED
Cryptanalysis of data/text20 = FAILED
Cryptanalysis of data/text21 = FAILED
Cryptanalysis of data/text22 = FAILED
Cryptanalysis of data/text23 = FAILED
Cryptanalysis of data/text24 = FAILED
Cryptanalysis of data/text25 = SUCCESS
Cryptanalysis of data/text26 = FAILED
Cryptanalysis of data/text27 = FAILED
Cryptanalysis of data/text28 = FAILED
Cryptanalysis of data/text29 = FAILED
Cryptanalysis of data/text30 = FAILED
Cryptanalysis of data/text31 = FAILED
Cryptanalysis of data/text32 = FAILED
Cryptanalysis of data/text33 = FAILED
Cryptanalysis of data/text34 = FAILED
Cryptanalysis of data/text35 = FAILED
Cryptanalysis of data/text36 = FAILED
Cryptanalysis of data/text37 = FAILED
Cryptanalysis of data/text38 = FAILED
Cryptanalysis of data/text39 = SUCCESS
Cryptanalysis of data/text40 = SUCCESS
Cryptanalysis of data/text41 = FAILED
Cryptanalysis of data/text42 = FAILED
Cryptanalysis of data/text43 = FAILED
Cryptanalysis of data/text44 = FAILED
Cryptanalysis of data/text45 = FAILED
Cryptanalysis of data/text46 = SUCCESS
Cryptanalysis of data/text47 = FAILED
Cryptanalysis of data/text48 = FAILED
Cryptanalysis of data/text49 = FAILED
Cryptanalysis of data/text50 = SUCCESS
Cryptanalysis of data/text51 = SUCCESS
Cryptanalysis of data/text52 = FAILED
Cryptanalysis of data/text53 = FAILED
Cryptanalysis of data/text54 = FAILED
Cryptanalysis of data/text55 = FAILED
Cryptanalysis of data/text56 = FAILED
Cryptanalysis of data/text57 = FAILED
Cryptanalysis of data/text58 = FAILED
Cryptanalysis of data/text59 = FAILED
Cryptanalysis of data/text60 = SUCCESS
Cryptanalysis of data/text61 = SUCCESS
Cryptanalysis of data/text62 = FAILED
Cryptanalysis of data/text63 = FAILED
Cryptanalysis of data/text64 = FAILED
Cryptanalysis of data/text65 = FAILED
Cryptanalysis of data/text66 = FAILED
Cryptanalysis of data/text67 = SUCCESS
Cryptanalysis of data/text68 = SUCCESS
Cryptanalysis of data/text69 = FAILED
Cryptanalysis of data/text70 = FAILED
Cryptanalysis of data/text71 = FAILED
Cryptanalysis of data/text72 = FAILED
Cryptanalysis of data/text73 = SUCCESS
Cryptanalysis of data/text74 = FAILED
Cryptanalysis of data/text75 = FAILED
Cryptanalysis of data/text76 = FAILED
Cryptanalysis of data/text77 = FAILED
Cryptanalysis of data/text78 = FAILED
Cryptanalysis of data/text79 = FAILED
Cryptanalysis of data/text80 = FAILED
Cryptanalysis of data/text81 = FAILED
Cryptanalysis of data/text82 = FAILED
Cryptanalysis of data/text83 = FAILED
Cryptanalysis of data/text84 = FAILED
Cryptanalysis of data/text85 = FAILED
Cryptanalysis of data/text86 = FAILED
Cryptanalysis of data/text87 = FAILED
Cryptanalysis of data/text88 = FAILED
Cryptanalysis of data/text89 = FAILED
Cryptanalysis of data/text90 = FAILED
Cryptanalysis of data/text91 = FAILED
Cryptanalysis of data/text92 = FAILED
Cryptanalysis of data/text93 = FAILED
Cryptanalysis of data/text94 = FAILED
Cryptanalysis of data/text95 = FAILED
Cryptanalysis of data/text96 = FAILED
Cryptanalysis of data/text97 = FAILED
Cryptanalysis of data/text98 = FAILED
Cryptanalysis of data/text99 = FAILED
Cryptanalysis of data/text100 = FAILED

18 texts successfully unciphered.
Test cryptanalyse_v1 : OK


----------------------------------------------


Cette méthode de décryptage, consistant à considérer que la lettre qui apparaît le plus représente la clef de chiffrement de 'E' et à traiter chaque colonne en évaluant la fréquence d'apparition des lettres, nous permet de déchiffrer 18 des 100 textes proposés.

18% d'efficacité, c'est peu, et la faiblesse de cette méthode vient de sa simplicité. En fait, il est assez facile de rendre un texte difficile à décrypter avec cette méthode, avec une clef plutôt de grande taille (comme pour un mot de passe : plus il est long et plus il est difficile d'évaluer toutes les combinaisons possibles pour le trouver) et il peut y avoir des indices de coïncidence intéressants sans pour autant que cela garantisse que la longueur de clef trouvée soit correcte.

Dans le cas général, les clés plus longues rendent l'analyse plus difficile car les répétitions sont moins visibles et les distributions des lettres sont différentes dans chaque texte, ce qui rend la tâche compliquée. La méthode de cryptanalyse utilisée repose sur l'indice de coïncidence (IC) et la fréquence des lettres, ce qui peut ne pas être suffisant pour tous les textes. Par exemple, si le texte chiffré est court, l'algorithme risque de rencontrer des difficultés pour estimer correctement les décalages. L'échec de 82%, montre que cette approche basique de cryptanalyse Vigenère est encore incomplète et doit être améliorée.


Question 12 :

Après lancement du test 7, l'affichage est le suivant :

----------------------------------------------


Test 7 : Cryptanalyse V2
---------------------
Test cryptanalyse_v2
Cryptanalysis of data/text1 = SUCCESS
Cryptanalysis of data/text2 = SUCCESS
Cryptanalysis of data/text3 = SUCCESS
Cryptanalysis of data/text4 = SUCCESS
Cryptanalysis of data/text5 = FAILED
Cryptanalysis of data/text6 = FAILED
Cryptanalysis of data/text7 = SUCCESS
Cryptanalysis of data/text8 = SUCCESS
Cryptanalysis of data/text9 = FAILED
Cryptanalysis of data/text10 = SUCCESS
Cryptanalysis of data/text11 = FAILED
Cryptanalysis of data/text12 = FAILED
Cryptanalysis of data/text13 = FAILED
Cryptanalysis of data/text14 = SUCCESS
Cryptanalysis of data/text15 = SUCCESS
Cryptanalysis of data/text16 = FAILED
Cryptanalysis of data/text17 = SUCCESS
Cryptanalysis of data/text18 = SUCCESS
Cryptanalysis of data/text19 = SUCCESS
Cryptanalysis of data/text20 = SUCCESS
Cryptanalysis of data/text21 = SUCCESS
Cryptanalysis of data/text22 = FAILED
Cryptanalysis of data/text23 = SUCCESS
Cryptanalysis of data/text24 = FAILED
Cryptanalysis of data/text25 = SUCCESS
Cryptanalysis of data/text26 = FAILED
Cryptanalysis of data/text27 = FAILED
Cryptanalysis of data/text28 = SUCCESS
Cryptanalysis of data/text29 = SUCCESS
Cryptanalysis of data/text30 = SUCCESS
Cryptanalysis of data/text31 = FAILED
Cryptanalysis of data/text32 = FAILED
Cryptanalysis of data/text33 = SUCCESS
Cryptanalysis of data/text34 = FAILED
Cryptanalysis of data/text35 = FAILED
Cryptanalysis of data/text36 = FAILED
Cryptanalysis of data/text37 = FAILED
Cryptanalysis of data/text38 = FAILED
Cryptanalysis of data/text39 = SUCCESS
Cryptanalysis of data/text40 = SUCCESS
Cryptanalysis of data/text41 = FAILED
Cryptanalysis of data/text42 = FAILED
Cryptanalysis of data/text43 = SUCCESS
Cryptanalysis of data/text44 = SUCCESS
Cryptanalysis of data/text45 = SUCCESS
Cryptanalysis of data/text46 = SUCCESS
Cryptanalysis of data/text47 = FAILED
Cryptanalysis of data/text48 = FAILED
Cryptanalysis of data/text49 = FAILED
Cryptanalysis of data/text50 = SUCCESS
Cryptanalysis of data/text51 = SUCCESS
Cryptanalysis of data/text52 = FAILED
Cryptanalysis of data/text53 = FAILED
Cryptanalysis of data/text54 = FAILED
Cryptanalysis of data/text55 = SUCCESS
Cryptanalysis of data/text56 = FAILED
Cryptanalysis of data/text57 = FAILED
Cryptanalysis of data/text58 = FAILED
Cryptanalysis of data/text59 = FAILED
Cryptanalysis of data/text60 = SUCCESS
Cryptanalysis of data/text61 = SUCCESS
Cryptanalysis of data/text62 = FAILED
Cryptanalysis of data/text63 = SUCCESS
Cryptanalysis of data/text64 = SUCCESS
Cryptanalysis of data/text65 = FAILED
Cryptanalysis of data/text66 = FAILED
Cryptanalysis of data/text67 = SUCCESS
Cryptanalysis of data/text68 = SUCCESS
Cryptanalysis of data/text69 = FAILED
Cryptanalysis of data/text70 = SUCCESS
Cryptanalysis of data/text71 = FAILED
Cryptanalysis of data/text72 = FAILED
Cryptanalysis of data/text73 = SUCCESS
Cryptanalysis of data/text74 = SUCCESS
Cryptanalysis of data/text75 = SUCCESS
Cryptanalysis of data/text76 = SUCCESS
Cryptanalysis of data/text77 = FAILED
Cryptanalysis of data/text78 = FAILED
Cryptanalysis of data/text79 = FAILED
Cryptanalysis of data/text80 = SUCCESS
Cryptanalysis of data/text81 = FAILED
Cryptanalysis of data/text82 = FAILED
Cryptanalysis of data/text83 = FAILED
Cryptanalysis of data/text84 = SUCCESS
Cryptanalysis of data/text85 = FAILED
Cryptanalysis of data/text86 = FAILED
Cryptanalysis of data/text87 = FAILED
Cryptanalysis of data/text88 = FAILED
Cryptanalysis of data/text89 = FAILED
Cryptanalysis of data/text90 = SUCCESS
Cryptanalysis of data/text91 = FAILED
Cryptanalysis of data/text92 = FAILED
Cryptanalysis of data/text93 = FAILED
Cryptanalysis of data/text94 = FAILED
Cryptanalysis of data/text95 = FAILED
Cryptanalysis of data/text96 = FAILED
Cryptanalysis of data/text97 = FAILED
Cryptanalysis of data/text98 = FAILED
Cryptanalysis of data/text99 = FAILED
Cryptanalysis of data/text100 = FAILED

43 texts successfully unciphered.
Test cryptanalyse_v2 : OK


----------------------------------------------


Nous pensons que cette cryptanalyse est plus efficace car il y a une amélioration importante par rapport au simple décalage des colonnes selon la recherche de la clé de chiffrement de 'E' de la première méthode de cryptanalyse.

L'utilisation de l'indice de coïncidence mutuelle pour calculer les décalages relatifs entre les colonnes est une meilleure approche, dans le sens où dans le chiffrement de Vigenère, chaque colonne du texte chiffré est décalée par un certain nombre de positions en fonction de la clé et en maximisant l'ICM pour chaque colonne par rapport à la première colonne, l'algorithme affine les décalages de manière beaucoup plus précise, permettant une meilleure reconstruction du texte.

Donc ici, l'amélioration importante de cette version est l'analyse statistique pour déterminer les décalages relatifs. Les 43 textes correctement cryptanalysés montrent que l'algorithme fonctionne partiellement, mais qu'il doit être perfectionné pour gérer une plus grande variété de textes. Donc cette fois, la longueur du texte est le critère principal pour pouvoir le déchiffrer et un texte trop court diminue simplement la fiabilité des analyses de coïcidence mutuelles entre colonnes (une fois que la longueur de clé a été déterminée) ce qui rend l'analyse de petits textes difficile et incertaine, ce qui peut expliquer nos résultats, selon nous, après avoir observé la longueur des textes (.cipher) et les mots clés de chiffrement.


Question 15 :

Après lancement du test 9, l'affichage est le suivant :

----------------------------------------------


Test 9 : Cryptanalyse V3
---------------------
Test cryptanalyse_v3
Cryptanalysis of data/text1 = SUCCESS
Cryptanalysis of data/text2 = SUCCESS
Cryptanalysis of data/text3 = SUCCESS
Cryptanalysis of data/text4 = SUCCESS
Cryptanalysis of data/text5 = SUCCESS
Cryptanalysis of data/text6 = SUCCESS
Cryptanalysis of data/text7 = SUCCESS
Cryptanalysis of data/text8 = SUCCESS
Cryptanalysis of data/text9 = SUCCESS
Cryptanalysis of data/text10 = SUCCESS
Cryptanalysis of data/text11 = SUCCESS
Cryptanalysis of data/text12 = SUCCESS
Cryptanalysis of data/text13 = SUCCESS
Cryptanalysis of data/text14 = SUCCESS
Cryptanalysis of data/text15 = SUCCESS
Cryptanalysis of data/text16 = SUCCESS
Cryptanalysis of data/text17 = SUCCESS
Cryptanalysis of data/text18 = SUCCESS
Cryptanalysis of data/text19 = SUCCESS
Cryptanalysis of data/text20 = SUCCESS
Cryptanalysis of data/text21 = SUCCESS
Cryptanalysis of data/text22 = SUCCESS
Cryptanalysis of data/text23 = SUCCESS
Cryptanalysis of data/text24 = SUCCESS
Cryptanalysis of data/text25 = SUCCESS
Cryptanalysis of data/text26 = SUCCESS
Cryptanalysis of data/text27 = SUCCESS
Cryptanalysis of data/text28 = SUCCESS
Cryptanalysis of data/text29 = SUCCESS
Cryptanalysis of data/text30 = SUCCESS
Cryptanalysis of data/text31 = SUCCESS
Cryptanalysis of data/text32 = SUCCESS
Cryptanalysis of data/text33 = SUCCESS
Cryptanalysis of data/text34 = SUCCESS
Cryptanalysis of data/text35 = SUCCESS
Cryptanalysis of data/text36 = SUCCESS
Cryptanalysis of data/text37 = SUCCESS
Cryptanalysis of data/text38 = SUCCESS
Cryptanalysis of data/text39 = SUCCESS
Cryptanalysis of data/text40 = SUCCESS
Cryptanalysis of data/text41 = SUCCESS
Cryptanalysis of data/text42 = SUCCESS
Cryptanalysis of data/text43 = SUCCESS
Cryptanalysis of data/text44 = SUCCESS
Cryptanalysis of data/text45 = SUCCESS
Cryptanalysis of data/text46 = SUCCESS
Cryptanalysis of data/text47 = SUCCESS
Cryptanalysis of data/text48 = SUCCESS
Cryptanalysis of data/text49 = SUCCESS
Cryptanalysis of data/text50 = SUCCESS
Cryptanalysis of data/text51 = SUCCESS
Cryptanalysis of data/text52 = SUCCESS
Cryptanalysis of data/text53 = SUCCESS
Cryptanalysis of data/text54 = SUCCESS
Cryptanalysis of data/text55 = SUCCESS
Cryptanalysis of data/text56 = SUCCESS
Cryptanalysis of data/text57 = SUCCESS
Cryptanalysis of data/text58 = SUCCESS
Cryptanalysis of data/text59 = SUCCESS
Cryptanalysis of data/text60 = SUCCESS
Cryptanalysis of data/text61 = SUCCESS
Cryptanalysis of data/text62 = SUCCESS
Cryptanalysis of data/text63 = SUCCESS
Cryptanalysis of data/text64 = SUCCESS
Cryptanalysis of data/text65 = SUCCESS
Cryptanalysis of data/text66 = SUCCESS
Cryptanalysis of data/text67 = SUCCESS
Cryptanalysis of data/text68 = SUCCESS
Cryptanalysis of data/text69 = SUCCESS
Cryptanalysis of data/text70 = SUCCESS
Cryptanalysis of data/text71 = SUCCESS
Cryptanalysis of data/text72 = SUCCESS
Cryptanalysis of data/text73 = SUCCESS
Cryptanalysis of data/text74 = SUCCESS
Cryptanalysis of data/text75 = SUCCESS
Cryptanalysis of data/text76 = SUCCESS
Cryptanalysis of data/text77 = SUCCESS
Cryptanalysis of data/text78 = SUCCESS
Cryptanalysis of data/text79 = SUCCESS
Cryptanalysis of data/text80 = SUCCESS
Cryptanalysis of data/text81 = FAILED
Cryptanalysis of data/text82 = SUCCESS
Cryptanalysis of data/text83 = SUCCESS
Cryptanalysis of data/text84 = SUCCESS
Cryptanalysis of data/text85 = SUCCESS
Cryptanalysis of data/text86 = FAILED
Cryptanalysis of data/text87 = SUCCESS
Cryptanalysis of data/text88 = FAILED
Cryptanalysis of data/text89 = FAILED
Cryptanalysis of data/text90 = SUCCESS
Cryptanalysis of data/text91 = SUCCESS
Cryptanalysis of data/text92 = SUCCESS
Cryptanalysis of data/text93 = SUCCESS
Cryptanalysis of data/text94 = FAILED
Cryptanalysis of data/text95 = SUCCESS
Cryptanalysis of data/text96 = FAILED
Cryptanalysis of data/text97 = SUCCESS
Cryptanalysis of data/text98 = SUCCESS
Cryptanalysis of data/text99 = SUCCESS
Cryptanalysis of data/text100 = SUCCESS

94 texts successfully unciphered.
Test cryptanalyse_v3 : OK


----------------------------------------------


Bien que cette méthode soit plus efficace (94 textes déchiffrés sur 100), elle présente un coût en temps d'exécution plus élevé, avec environ 9 secondes sur les essais contre moins de 0.5 seconde pour les versions précédentes, bien que celles-ci soient moins performantes sur les résultats.

Nous remarquons cependant que certains tests persistent à échouer, malgré l'efficacité de cette méthode qui effectue des décalages circulaires afin de tester toutes les combinaisons des fréquences d'apparition de chaque lettre en les comparant directement avec les fréquences obtenues du texte Germinal de Zola (texte suffisamment long pour être représentatif de la fréquence d'apparition de chaque lettre dans la langue Française), et qui conserve ensuite pour chaque colonne, la permutation trouvée qui corrèle le plus avec les fréquences observées dans ce texte Français, afin d'identifier chaque lettre en question.

Il semblerait, après observation, que les textes qui restent non-déchiffrés avec cette version de l'algorithme soient trop courts et/ou ne reflètent pas les fréquences d'apparition des lettres habituelles de la langue. Donc en somme, les textes courts, atypiques ou avec une faible diversité de lettres et un vocabulaire simple, ainsi qu'avec une clef assez longue sont plus difficiles à déchiffrer par l'algorithme. Par exemple, le texte avec la clé la plus courte n'ayant pas été déchiffré est le texte 89 chiffré à l'aide d'une clé de longueur 9, tandis que l'algorithme a réussi à identifier toutes les clés de taille inférieure. 

Les clés plus courtes entraînent une répétition plus fréquente du schéma de chiffrement, ce qui rend l'identification de la clé et de sa longueur plus facile et qui diminue le nombre de décalages à identifier (car les clés sont plus courtes) et à la fois réduit les possibilités de commettre des erreurs.
