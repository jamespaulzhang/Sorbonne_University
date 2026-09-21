public abstract class NonCombattant extends Heros{
	private int intelligence;
	public NonCombattant(String prenom,int intelligence){
		super(prenom);
		this.intelligence = intelligence;
	}

	public int getIntelligence(){
		return intelligence;
	}
	
	public abstract void action();
}