/**
 * Yuxiang ZHANG , numero etudiant : 21202829
 * Antoine LECOMTE, numero etudiant : 21103457
 * Groupe 4 
 */

/**
 * La classe Joyau représente un contenu de type Joyau dans la simulation.
 * Elle hérite de la classe Contenu.
 */
public class Joyau extends Contenu {
    private String nom;
    private int prix;
    
    /**
     * Constructeur de la classe Joyau.
     * @param nom Le nom du Joyau.
     * @param prix Le prix du Joyau.
    On ne peut pas utiliser les accesseurs ci-dessous si on n'ajoute pas les initialisation suivantes 
    (l'appel au constructeur de la classe Contenu n'est pas suffisant).
    */
    public Joyau(String nom, int prix) {
        super(nom, prix);
        this.nom = nom;
        this.prix = prix;
    }

    /**
     * Obtient le nom du Joyau.
     * @return Le nom du Joyau.
     */
    public String getNom() {
        return nom;
    }

    /**
     * Obtient le prix du Joyau.
     * @return Le prix du Joyau.
     */
    public int getPrix() {
        return prix;
    }
}
