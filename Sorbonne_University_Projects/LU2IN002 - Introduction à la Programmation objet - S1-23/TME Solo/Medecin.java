public class Medecin extends NonCombattant{

	public Medecin(String prenom,int intelligence){
		super(prenom,intelligence);
	}

	public String toString(){
		return getPrenom() + " intellifence : " + getIntelligence() + ", classe : Medecin";
	}

	public void action(){
		int nb_soignes = (int)getIntelligence() / 2;
		System.out.println(toString() + ", soigne " + nb_soignes + " heros");
	}
}