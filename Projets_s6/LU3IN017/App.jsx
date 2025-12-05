import { useState } from "react";
import Livre from "./Livre.jsx";

function App(){
  const dateAuj = new Date();
  const dateE = new Date(2023, 1, 19);
  const [titreCollection, setTitreCollection] = useState("");
  const [livres, setLivres] = useState([{
    auteur: "Hugo Victor", titre: "La Légende des siècles", emprunt: {statut: false,
    dateEmprunt:dateAuj}, cote: "HUG001"},
    {auteur: "Hugo Victor", titre: "Les Misérables", emprunt: {statut: false,
    dateEmprunt:dateAuj}, cote: "HUG002"},
    {auteur: "Zola Émile", titre: "L'Assommoir", emprunt:
    {statut: true, dateEmprunt:dateE}, cote: "ZOL001"}]);

    const changeTitre = (evt) => setTitreCollection(evt.target.value);
    const addLivre = (evt) =>{evt.preventDefault();
    const newLivre = {
      auteur: evt.target.chp_auteur.value,
      titre: evt.target.chp_titre.value,
      emprunt: { statut: false, dateEmprunt: dateAuj },
      cote: evt.target.chp_cote.value
    };
      setLivres([...livres, newLivre]);
      evt.target.reset();
    };
  
    return (
      <div>
      <h1>{titreCollection}</h1>
      <label htmlFor="chp_collec">Titre de la collection ?</label>
      <input id="chp_collec" onChange={changeTitre}/>
      
      <ul>
      {livres.map((livre) => <Livre key={livre.cote} auteur={livre.auteur}
      titre={livre.titre} cote={livre.cote} emprunt={livre.emprunt}/>)}
      </ul>

      <form onSubmit={addLivre}>
      <label htmlFor="chp_titre" name="chp_titre">Titre</label>
      <input id="chp_titre" />
      <br/>
      <label htmlFor="chp_auteur">Auteur</label>
      <input id="chp_auteur" name="chp_auteur" placeholder="Nom Prénom" />
      <br/>
      <label htmlFor="chp_cote">Cote</label>
      <input id="chp_cote" name="chp_cote" placeholder="INIxxx"/>
      <br/>
      <button type="submit">Ajouter un livre</button>
      </form>
      </div>
    );
  }

export default App;