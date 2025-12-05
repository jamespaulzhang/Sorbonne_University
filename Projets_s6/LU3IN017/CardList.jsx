import { useState } from 'react';
import Card from './Card.jsx';

function CardList() {
    const [cartes, setCartes] = useState([
        { card: 'toto', display: 'visible' },
        { card: 'tata', display: 'hidden' },
        { card: 'titi', display: 'hidden' },
        { card: 'tutu', display: 'hidden' }
    ]);

    return (
        <div className="cardlist">
            {cartes.map((card, index) => (
                <Card key={index} symbol={card.card} affichage={card.display} />
            ))}
        </div>
    );
}

export default CardList;
