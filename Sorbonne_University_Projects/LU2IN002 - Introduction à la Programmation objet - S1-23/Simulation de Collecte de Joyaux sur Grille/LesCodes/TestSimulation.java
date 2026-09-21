/**
 * Yuxiang ZHANG , numero etudiant : 21202829
 * Antoine LECOMTE, numero etudiant : 21103457
 * Groupe 4
 */

 /**
 * La classe TestSimulation est utilisée pour tester la simulation du jeu.
 */
public class TestSimulation {
    public static void main(String[] args) {
        // Créer une instance de Simulation avec des paramètres appropriés
        Simulation simulation = new Simulation(10, 10, 20, 3, 0, 0);

        // Exécuter la simulation avec un nombre spécifié d'étapes
        simulation.lance(20);

        // Récupérer la grille et l'agent de la simulation
        Grille grille = simulation.getGrille();
        Agent4 agent = simulation.getAgent4();

        // Afficher les informations finales de la simulation
        System.out.println("Simulation terminée. Informations finales :\n");
        System.out.println("Agent position : (" + agent.getPositionX() + ", " + agent.getPositionY() + ")");
        System.out.println("Agent fortune : " + agent.fortune() + " pièces d'or");
        agent.contenuSac();

        // Afficher le contenu final de la grille
        System.out.println("\nContenu final de la grille :\n" + grille);
    }
}