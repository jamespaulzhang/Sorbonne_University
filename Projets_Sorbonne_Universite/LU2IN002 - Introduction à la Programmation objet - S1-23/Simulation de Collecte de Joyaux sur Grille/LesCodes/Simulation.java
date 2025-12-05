/**
 * Yuxiang ZHANG, numero etudiant: 21202829
 * Antoine LECOMTE, numero etudiant: 21103457
 * Groupe 4
 */

/**
 * La classe Simulation représente le déroulement de la simulation du jeu.
 */
public class Simulation {
    private Agent4 agent;
    private Grille grille;
    private Contenu[] contenus;

    /**
     * Initialise une nouvelle simulation avec les paramètres spécifiés.
     * 
     * @param nbLig          Nombre de lignes de la grille.
     * @param nbCol          Nombre de colonnes de la grille.
     * @param numContenus    Nombre total de contenus à générer.
     * @param nbGardiens     Nombre de gardiens à générer.
     * @param agentInitialX  Position initiale en abscisse de l'agent.
     * @param agentInitialY  Position initiale en ordonnée de l'agent.
     */
    public Simulation(int nbLig, int nbCol, int numContenus, int nbGardiens, int agentInitialX, int agentInitialY) {
        this.grille = new Grille(nbLig, nbCol);
        this.agent = new Agent4(agentInitialX, agentInitialY, this.grille);
        this.contenus = new Contenu[numContenus];

        if (numContenus > nbLig * nbCol) {
            System.out.println("Impossible de générer " + numContenus + " contenus dans une grille de " + nbLig + " lignes et " + nbCol + " colonnes.");
            return;
        }

        int contenusGeneres = 0;
        int gardiensGeneres = 0;
        int joyauxGeneres = 0;

        // Génération des gardiens
        while (gardiensGeneres < nbGardiens) {
            int x = (int) (Math.random() * nbLig);
            int y = (int) (Math.random() * nbCol);

            try {
                if (grille.caseEstVide(x, y)) {
                    int pointsDeVieGardien = (int) (Math.random() * 201);
                    Gardien gardien = new Gardien("Gardien", pointsDeVieGardien, this.grille);
                    grille.setCase(x, y, gardien);
                    contenus[contenusGeneres] = gardien;
                    contenusGeneres++;
                    gardiensGeneres++;
                }
            } catch (CoordonneesIncorrectesException e) {
                System.out.println(e.getMessage());
            }
        }

        // Génération des joyaux
        while (joyauxGeneres + gardiensGeneres < numContenus) {
            int x = (int) (Math.random() * nbLig);
            int y = (int) (Math.random() * nbCol);

            try {
                if (grille.caseEstVide(x, y)) {
                    int prix = (int) (Math.random() * 4000 + 1);
                    float randomType = (float) Math.random();

                    Joyau joyau;
                    try {
                        if (randomType < 1.0 / 3) {
                            joyau = new Joyau("Diamant", prix);
                        } else if (randomType < 2.0 / 3) {
                            joyau = new Joyau("Opale", prix);
                        } else {
                            joyau = new Joyau("Rubis", prix);
                        }

                        grille.setCase(x, y, joyau);
                        contenus[contenusGeneres] = joyau;
                        contenusGeneres++;
                        joyauxGeneres++;
                    } catch (CoordonneesIncorrectesException e) {
                        System.out.println(e.getMessage());
                    }
                }
            } catch (CoordonneesIncorrectesException e) {
                System.out.println(e.getMessage());
            }
        }
    }

    /**
     * Retourne une représentation textuelle de l'état actuel de la simulation.
     * 
     * @return Une chaîne de caractères représentant l'état de la simulation.
     */
    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("État de la simulation :\n");
        result.append("Grille :\n").append(grille).append("\n");
        result.append("Agent :\n").append("Position : (").append(agent.getPositionX()).append(", ").append(agent.getPositionY()).append(")\n");
        result.append("Fortune : ").append(agent.fortune()).append("\n");
        return result.toString();
    }

    /**
     * Lance la simulation pour un nombre spécifié d'étapes.
     * 
     * @param nbEtapes Le nombre d'étapes de la simulation.
     */
    public void lance(int nbEtapes) {
        for (int i = 0; i < nbEtapes; i++) {
            System.out.println("Étape " + (i + 1) + " :\n");
    
            int xNew = agent.getPositionX() + (int)(Math.random() * 3 - 1);
            int yNew = agent.getPositionY() + (int)(Math.random() * 3 - 1);
            boolean avecForce = Math.random() < 0.3;
            int force = avecForce ? (int)(Math.random() * 91 + 10) : 0;
    
            try {
                if (avecForce) {
                    agent.seDeplacer(xNew, yNew, force);
                } else {
                    agent.seDeplacer(xNew, yNew);
                }
            } catch (DeplacementIncorrectException e) {
                System.out.println("Erreur de déplacement : " + e.getMessage());
            }
    
            // Déplacement des gardiens
            for (Contenu contenu : contenus) {
                if (contenu instanceof Gardien) {
                    Gardien gardien = (Gardien) contenu;
                    int newX = gardien.getPositionX() + (int)(Math.random() * 3 - 1);
                    int newY = gardien.getPositionY() + (int)(Math.random() * 3 - 1);
    
                    try {
                        gardien.seDeplacer(newX, newY);
                    } catch (DeplacementIncorrectException ignored) {
                        // Pour ne pas rendre illisible l'affichage à l'exécution, on n'affiche rien pour l'exception.
                    }
                }
            }
    
            System.out.println("Informations sur l'étape :\n");
            System.out.println(this);
    
            try {
                Thread.sleep(1000); // attendre 1 seconde pour avoir le temps de voir les changements dans la grille à l'exécution.
            } catch (InterruptedException e) {
                e.getMessage();
            }
    
            grille.affiche(10); // ajuster la taille pour l'affichage de la grille
        }
    }

    /**
     * Retourne la grille de la simulation.
     * 
     * @return La grille de la simulation.
     */
    public Grille getGrille(){
        return grille;
    }

    /**
     * Retourne l'agent de la simulation.
     * 
     * @return L'agent de la simulation.
     */
    public Agent4 getAgent4(){
        return agent;
    }
}
