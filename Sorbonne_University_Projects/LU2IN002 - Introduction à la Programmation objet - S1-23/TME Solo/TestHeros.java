public class TestHeros{
	public static void main(String[] args){
		Guilde g = new Guilde(10);
		Guerrier g1 = new Guerrier("William",8,"epee");
		Guerrier g2 = new Guerrier("Astrid",7,"marteau");

		g.ajouterHeros(g1);
		g.ajouterHeros(g2);
		System.out.println(g);
		g.actionGuilde();

		Medecin m = new Medecin("Xun",9);
		g.ajouterHeros(m);
		System.out.println();
		System.out.println(g);
		g.actionGuilde();

	}
}