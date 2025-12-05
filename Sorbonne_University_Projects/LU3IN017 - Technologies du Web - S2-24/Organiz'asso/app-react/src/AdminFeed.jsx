import {useState, useEffect} from "react";
import {getMessagesRequest, addMessageRequest, removeMessageRequest} from './api/messagesAPI.jsx';
import MessageList from "./MessageList.jsx";
import AddMessage from "./AddMessage.jsx";

function AdminFeed (props) {
    const [messages, setMessages] = useState([]);

    useEffect(() => {
        getMessagesRequest(setMessages, "", "1");
    }, []);

    const addMessage = (event) => {
        addMessageRequest(event, props, messages, setMessages, true);
    };

    const removeMessage = (mid) => {
        removeMessageRequest(messages, setMessages, mid);
    };

    return (
        <main>
            <div className="box">
                <h2>Nouveau message</h2>
                <AddMessage addMessage={addMessage} />
            </div>
            <div className="box">
                <h1>Derniers messages publiés par les utilisateurs</h1>
                <article id="feed">
                    <MessageList messages={messages} removeMessage={removeMessage} currentUser={props.currentUser} toUserPage={props.toUserPage} />
                </article>
            </div>
        </main>
    );
}

export default AdminFeed;