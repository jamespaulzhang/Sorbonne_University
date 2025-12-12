public class Guerrier extends Combattant{
	private String arme;
	public Guerrier(String prenom,int force,String arme){
		super(prenom,force);
		this.arme = arme;
	}

	public String toString(){
		return getPrenom() + " force : " + getForce() + ", classe : guerrier, arme : " + arme;
	}

	public void action(){
		int degats;
		if (arme == "marteau"){
			degats = 2 * getForce();
		}else{
			degats = getForce() + (int)(Math.random()*6 + 1);
		}
		System.out.println(toString() + ", attaque pour " + degats + " degats"); 
	}


}