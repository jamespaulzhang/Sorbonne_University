function App(){
    const dateAuj = new Date();
    const dateE = new Date(2013, 7, 19);
    const [livres, setLivres] = useState([titreCollection || livres]);
  
    return <div>
      <ul>
        {livres.map((livre) => 
        <Livre auteur={livre.auteur} 
        titre={livre.titre} 
        cote={livre.cote} 
        emprunt={livre.emprunt}/>)}
      </ul>
    </div>
  }