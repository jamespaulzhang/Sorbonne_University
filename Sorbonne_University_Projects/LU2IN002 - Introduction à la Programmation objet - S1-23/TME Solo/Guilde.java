public class Guilde{
	private Heros[] tab;
	private int nb_heros;

	public Guilde(int taille){
		tab = new Heros[taille];
	}

	public void ajouterHeros(Heros h){
		if (nb_heros < tab.length){
			tab[nb_heros++] = h;
		}else{
			System.out.println("Le tableau est plèine,pas possible d'ajouter un héro");
		}
	}

	public void actionGuilde(){
		for(int i = 0;i < tab.length;i++){
			if (tab[i] != null){
				tab[i].action();
			}
		}
	}

	public String toString(){
		String s = "La guilde est composée de : \n";
		for(int i = 0;i < tab.length;i++){
			if (tab[i] != null){
				s += tab[i].toString() + "\n";
			}
		}
		return s;
	}
}