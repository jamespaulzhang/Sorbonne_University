{-# LANGUAGE TypeSynonymInstances, FlexibleInstances #-}
-- as_et_huits.hs
-- Modélisation du jeu "Les as et les huits" avec DEMO-S5

import Data.List
import Control.Monad (foldM_, foldM, when, unless)
import EREL (Erel, bl)
import DEMO_S5

-- Noms du agents
playerA = Ag 0  -- Joueur A
playerB = Ag 1  -- Joueur B  
playerC = Ag 2  -- Joueur C

-- Types de mains possibles
data HandType = AA | A8 | H88 deriving (Eq, Ord, Show)

-- Fonction pour mapper une main d'un agent à une proposition atomique
handToProp :: Agent -> HandType -> Prp
handToProp (Ag 0) AA = P 0   -- joueur A a AA
handToProp (Ag 0) A8 = P 1   -- joueur A a A8
handToProp (Ag 0) H88 = P 2  -- joueur A a 88
handToProp (Ag 1) AA = P 3   -- joueur B a AA
handToProp (Ag 1) A8 = P 4   -- joueur B a A8
handToProp (Ag 1) H88 = P 5  -- joueur B a 88
handToProp (Ag 2) AA = P 6   -- joueur C a AA
handToProp (Ag 2) A8 = P 7   -- joueur C a A8
handToProp (Ag 2) H88 = P 8  -- joueur C a 88

-- Représentation d'un monde comme tuple de mains (A, B, C)
type World = (HandType, HandType, HandType)

-- Tous les mondes possibles (avec contraintes sur le nombre de cartes)
allWorlds :: [World]
allWorlds = 
  let allCombinations = [(x,y,z) | x <- [AA, A8, H88], 
                                   y <- [AA, A8, H88], 
                                   z <- [AA, A8, H88]]
  in filter isValidWorld allCombinations
  where
    -- Un monde est valide si le nombre total de cartes correspond
    -- (4 as et 4 huits au total)
    isValidWorld (ha, hb, hc) =
      let countAA = sum [if h == AA then 2 else if h == A8 then 1 else 0 | h <- [ha, hb, hc]]
          count88 = sum [if h == H88 then 2 else if h == A8 then 1 else 0 | h <- [ha, hb, hc]]
      in countAA <= 4 && count88 <= 4

-- Fonction pour obtenir la main d'un agent dans un monde
handOf :: Agent -> World -> HandType
handOf (Ag 0) (ha, _, _) = ha
handOf (Ag 1) (_, hb, _) = hb
handOf (Ag 2) (_, _, hc) = hc

-- Fonction pour obtenir les mondes équivalents pour un agent
-- (les mondes où les mains visibles sont les mêmes)
equivalenceClass :: Agent -> World -> [World]
equivalenceClass ag currentWorld =
  filter (\w -> sameVisibleCards ag currentWorld w) allWorlds
  where
    sameVisibleCards (Ag 0) (_, hb, hc) (_, hb', hc') = hb == hb' && hc == hc'
    sameVisibleCards (Ag 1) (ha, _, hc) (ha', _, hc') = ha == ha' && hc == hc'
    sameVisibleCards (Ag 2) (ha, hb, _) (ha', hb', _) = ha == ha' && hb == hb'

-- Fonction pour convertir un monde en index (pour le modèle)
worldToIndex :: [(World, Int)] -> World -> Int
worldToIndex mapping w = case lookup w mapping of
  Just i -> i
  Nothing -> error "World not found in mapping"

-- Mapping des mondes vers des indices
worldIndex :: [(World, Int)]
worldIndex = zip allWorlds [0..]

-- Fonction d'évaluation: assigne les propositions vraies dans chaque monde
valuation :: [(Int, [Prp])]
valuation = map (\(w, i) -> (i, propsForWorld w)) worldIndex
  where
    propsForWorld (ha, hb, hc) =
      [handToProp playerA ha, handToProp playerB hb, handToProp playerC hc]

-- Relations d'accessibilité pour chaque agent
accessibility :: [(Agent, Erel Int)]
accessibility = 
  [ (playerA, equivalenceForAgent playerA),
    (playerB, equivalenceForAgent playerB),
    (playerC, equivalenceForAgent playerC) ]
  where
    equivalenceForAgent ag =
      let classes = groupBy sameClass (sortOn key allWorlds)
          key w = case ag of
            (Ag 0) -> let (_, y, z) = w in (y, z)  -- B et C sont visibles
            (Ag 1) -> let (x, _, z) = w in (x, z)  -- A et C sont visibles
            (Ag 2) -> let (x, y, _) = w in (x, y)  -- A et B sont visibles
          sameClass w1 w2 = key w1 == key w2
          toIndices = map (worldToIndex worldIndex)
      in map toIndices classes

-- Construction du modèle épistémique initial
model0 :: EpistM Int
model0 = Mo 
  (map snd worldIndex)  -- tous les états (mondes)
  [playerA, playerB, playerC]  -- tous les agents
  valuation  -- fonction d'évaluation
  accessibility  -- relations d'accessibilité
  (map snd worldIndex)  -- mondes actuels (tous au début)

-- Fonction auxiliaire pour obtenir la liste des états d'un modèle
statesOf :: EpistM a -> [a]
statesOf (Mo s _ _ _ _) = s

-- Formule: l'agent sait quelle main il a
knowsHand :: Agent -> Form Int
knowsHand ag = 
  case ag of
    (Ag 0) -> Disj [Kn ag (Prp (P 0)), Kn ag (Prp (P 1)), Kn ag (Prp (P 2))]
    (Ag 1) -> Disj [Kn ag (Prp (P 3)), Kn ag (Prp (P 4)), Kn ag (Prp (P 5))]
    (Ag 2) -> Disj [Kn ag (Prp (P 6)), Kn ag (Prp (P 7)), Kn ag (Prp (P 8))]
    _ -> error "Agent inconnu"

-- Formule: l'agent ne sait pas quelle main il a
doesntKnow :: Agent -> Form Int
doesntKnow ag = Ng (knowsHand ag)

-- Générer l'ordre des annonces en fonction du joueur qui commence
generateOrder :: Agent -> [Agent]
generateOrder startPlayer =
  let infiniteOrder = cycle [playerA, playerB, playerC]
  in take 6 $ dropWhile (/= startPlayer) infiniteOrder

-- Simuler une partie complète
simulateGame :: World -> Agent -> IO ()
simulateGame realWorld startPlayer = do
  let order = generateOrder startPlayer
  putStrLn $ "\n=== Simulation avec monde réel : " ++ show realWorld ++ " ==="
  putStrLn $ "Ordre des annonces (départ " ++ show startPlayer ++ ") : " ++ show order
  putStrLn $ "Nombre initial de mondes possibles : " ++ show (length (statesOf model0))
  
  -- Obtenir l'indice du monde réel
  let realIndex = worldToIndex worldIndex realWorld
  
  -- Fonction récursive pour gérer la partie avec arrêt prématuré
  let go :: EpistM Int -> Int -> [Agent] -> IO ()
      go model roundCount [] = do
        putStrLn $ "\n=== Fin de la partie : échec après " ++ show roundCount ++ " tours ==="
        putStrLn $ "  Aucun joueur n'a pu déterminer ses cartes"
      go model roundCount (ag:agents) = do
        let currentRound = roundCount + 1
        putStrLn $ "\nTour " ++ show currentRound ++ " - Joueur " ++ show ag ++ " parle"
        
        -- Le joueur dit la vérité
        let knows = isTrueAt model realIndex (knowsHand ag)
        if knows
          then do
            putStrLn $ "  Le joueur " ++ show ag ++ " annonce qu'il sait sa main"
            -- Il annonce sa main exacte (pour l'information, même si le jeu s'arrête)
            let hand = handOf ag realWorld
            let announcement = case (ag, hand) of
                  ((Ag 0), AA) -> Kn ag (Prp (P 0))
                  ((Ag 0), A8) -> Kn ag (Prp (P 1))
                  ((Ag 0), H88) -> Kn ag (Prp (P 2))
                  ((Ag 1), AA) -> Kn ag (Prp (P 3))
                  ((Ag 1), A8) -> Kn ag (Prp (P 4))
                  ((Ag 1), H88) -> Kn ag (Prp (P 5))
                  ((Ag 2), AA) -> Kn ag (Prp (P 6))
                  ((Ag 2), A8) -> Kn ag (Prp (P 7))
                  ((Ag 2), H88) -> Kn ag (Prp (P 8))
            let newModel = upd_pa model announcement
            putStrLn $ "  Monde(s) restant(s) : " ++ show (length (statesOf newModel))
            putStrLn $ "  ✨ Le jeu se termine ici car " ++ show ag ++ " a annoncé qu'il sait sa main!"
            putStrLn $ "\n=== Résultat: " ++ show ag ++ " gagne au tour " ++ show currentRound ++ " ==="
          else do
            putStrLn $ "  Le joueur " ++ show ag ++ " annonce qu'il ne sait pas"
            let newModel = upd_pa model (doesntKnow ag)
            putStrLn $ "  Monde(s) restant(s) : " ++ show (length (statesOf newModel))
            go newModel currentRound agents
  
  -- Démarrer la simulation
  go model0 0 order

-- Tester plusieurs configurations de jeu
testMultipleGames :: IO ()
testMultipleGames = do
  putStrLn "Jeu des As et des Huits - Simulations multiples"
  putStrLn $ "=" ++ replicate 60 '='
  
  -- Afficher tous les mondes possibles
  putStrLn "\nMondes possibles initiaux :"
  mapM_ (putStrLn . ("  " ++) . show) allWorlds
  putStrLn $ "Total : " ++ show (length allWorlds) ++ " mondes"
  
  -- Liste des configurations à tester
  let configurations = [
        -- (monde réel, joueur qui commence)
        ((AA, A8, A8), playerA),      -- Partie 1
        ((H88, AA, A8), playerB),     -- Partie 2
        ((A8, A8, AA), playerC),      -- Partie 3
        ((AA, H88, A8), playerA),     -- Partie 4
        ((H88, A8, AA), playerB),     -- Partie 5
        ((A8, AA, H88), playerC),     -- Partie 6
        ((AA, AA, H88), playerA),     -- Partie 7
        ((H88, H88, AA), playerB),    -- Partie 8
        ((A8, H88, AA), playerC)      -- Partie 9
        ]
  
  -- Exécuter les simulations
  mapM_ (\(world, start) -> do
    putStrLn $ "\n" ++ replicate 60 '-'
    simulateGame world start
    ) configurations

-- Fonction principale
main :: IO ()
main = do
  putStrLn "Modélisation épistémique du jeu 'Les As et les Huits'"
  putStrLn "Avec la bibliothèque DEMO-S5"
  putStrLn ""
  
  testMultipleGames
  
  putStrLn $ "\n" ++ replicate 60 '='
  putStrLn "Toutes les simulations sont terminées."