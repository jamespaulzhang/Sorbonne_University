import {useState, useEffect} from "react";
import {getMessagesRequest, removeMessageRequest} from '../messages_request.jsx';
import MessageList from "../components/MessageList.jsx";


function Result (props) {

    const [messages, setMessages] = useState([]);

    useEffect(() => {
        getMessagesRequest(setMessages, "", (props.currentUser.admin ? "" : "0"), props.param.search, props.param.date_from, props.param.date_to);
    }, [props.param.search, props.param.date_from, props.param.date_to, props.currentUser.admin]);

    const removeMessage = (mid) => {
        removeMessageRequest(messages, setMessages, mid);
    };

    return (

        <main>
            <div id="back-to-index"><a onClick={props.toFeedPage}>❮ Retour à l'accueil</a></div>
            <div className="box">
                <h2>Résultat de votre recherche</h2>
                <article id="feed">
                    <MessageList messages={messages} removeMessage={removeMessage} currentUser={props.currentUser} toUserPage={props.toUserPage}/>
                </article>
            </div>
        </main>
    );

}

export default Result;