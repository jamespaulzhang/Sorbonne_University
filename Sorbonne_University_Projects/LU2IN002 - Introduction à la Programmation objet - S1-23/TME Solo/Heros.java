public abstract class Heros{
	public final int id;
	private String prenom;
	private static int compteur = 0;

	public Heros(String prenom){
		id = compteur++;
		this.prenom = prenom;
	}

	public String getPrenom(){
		return prenom;
	}

	public abstract void action();
}