import { useState } from "react";

function Card(props) {
    const [affichage, setAffichage] = useState(props.affichage || 'visible');

    function handleClick() {
        if (affichage === "visible") {
            setAffichage("hidden");
        } else {
            setAffichage("visible");
        }
    }

    return <button className="Card" onClick={handleClick}>
        {affichage === "visible" ? props.symbol : "-"}
        </button>;
}

export default Card;