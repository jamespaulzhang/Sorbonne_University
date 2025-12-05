import {useEffect, useRef} from "react";


function SearchBar (props) {

    const search = useRef(null);
    const date_from = useRef(null);
    const date_to = useRef(null);

    useEffect(() => {
        let today = new Date().toISOString().split('T')[0];
        date_from.current.max = today;
        date_to.current.max = today;
    }, []);

    const rechercher = () => {
        const search_ = search.current.value;
        const date_from_ = date_from.current.value;
        const date_to_ = date_to.current.value;
        if(search_ !== "" || date_from_ !== "" || date_to_ !== "")
            props.toResultPage(search_, date_from_, date_to_)
        search.current.value = date_from.current.value = date_to.current.value = "";
    }

    return (
        <form id="searchbar">
            <div>
                <label htmlFor="searchbar-search">Recherche</label>
                <input id="searchbar-search" ref={search} type="text" placeholder="Recherche"/>
            </div>
            <div>
                <label htmlFor="searchbar-from">Date de début</label>
                <input id="searchbar-from" ref={date_from} type="date" placeholder="De" max="today"/>
            </div>
            <div>
                <label htmlFor="searchbar-to">Date de fin</label>
                <input id="searchbar-to" ref={date_to} type="date" placeholder="À"/>
            </div>
            <button id="searchbar-btn" type="button" onClick={rechercher}>Rechercher</button>
        </form>
    );

}

export default SearchBar;