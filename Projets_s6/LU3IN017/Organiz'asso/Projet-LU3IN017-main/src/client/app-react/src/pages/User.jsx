import {useState, useEffect} from "react";
import axios from "axios";
import server_request from '../server_request.jsx';
import {getMessagesRequest, addMessageRequest, removeMessageRequest} from '../messages_request.jsx';
import MessageList from "../components/MessageList.jsx";
import AddMessage from "../components/AddMessage.jsx";


function User (props) {

    const [messages, setMessages] = useState([]);
    const [isAdmin, setIsAdmin] = useState(props.user.user.admin);

    useEffect(() => {
        getMessagesRequest(setMessages, props.user.user.uid, (props.currentUser.admin ? "" : "0"));
    }, []);

    const addMessage = (event) => {
        addMessageRequest(event, props, messages, setMessages, false);
    };

    const removeMessage = (mid) => {
        removeMessageRequest(messages, setMessages, mid);
    };

    const setAdmin = () => {
        if (props.user.user.uid !== props.currentUser.uid) {
            const status = !isAdmin;
            server_request(axios.patch, 'api/admin', {
                uid: props.user.user.uid,
                status: status
            }).then(setIsAdmin(status))
            .catch(console.error);
        }
    };

    return (
        <main>
            <div id="back-to-index"><a onClick={props.toFeedPage}>❮ Retour à l'accueil</a></div>

            <h1>Profil de {props.user.user.fname} {props.user.user.lname}

                {props.currentUser.admin ?

                    (props.user.user.uid === props.currentUser.uid) ?

                        <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" fill="currentColor" className="admin-icon bi bi-star-fill" viewBox="0 0 16 16">
                            <path
                                d="M3.612 15.443c-.386.198-.824-.149-.746-.592l.83-4.73L.173 6.765c-.329-.314-.158-.888.283-.95l4.898-.696L7.538.792c.197-.39.73-.39.927 0l2.184 4.327 4.898.696c.441.062.612.636.282.95l-3.522 3.356.83 4.73c.078.443-.36.79-.746.592L8 13.187l-4.389 2.256z"/>
                        </svg>

                        :

                        <button type="button" className="btn-icon-profil" onClick={setAdmin}>
                            {isAdmin ?
                                <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" fill="currentColor"
                                     className="admin-icon bi bi-star-fill" viewBox="0 0 16 16">
                                    <path
                                        d="M3.612 15.443c-.386.198-.824-.149-.746-.592l.83-4.73L.173 6.765c-.329-.314-.158-.888.283-.95l4.898-.696L7.538.792c.197-.39.73-.39.927 0l2.184 4.327 4.898.696c.441.062.612.636.282.95l-3.522 3.356.83 4.73c.078.443-.36.79-.746.592L8 13.187l-4.389 2.256z"/>
                                </svg>
                                :
                                <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" fill="currentColor"
                                     className="admin-icon bi bi-star" viewBox="0 0 16 16">
                                    <path
                                        d="M2.866 14.85c-.078.444.36.791.746.593l4.39-2.256 4.389 2.256c.386.198.824-.149.746-.592l-.83-4.73 3.522-3.356c.33-.314.16-.888-.282-.95l-4.898-.696L8.465.792a.513.513 0 0 0-.927 0L5.354 5.12l-4.898.696c-.441.062-.612.636-.283.95l3.523 3.356-.83 4.73zm4.905-2.767-3.686 1.894.694-3.957a.56.56 0 0 0-.163-.505L1.71 6.745l4.052-.576a.53.53 0 0 0 .393-.288L8 2.223l1.847 3.658a.53.53 0 0 0 .393.288l4.052.575-2.906 2.77a.56.56 0 0 0-.163.506l.694 3.957-3.686-1.894a.5.5 0 0 0-.461 0z"/>
                                </svg>
                            }
                        </button>
                : "" }
            </h1>

            {props.user.user.uid === props.currentUser.uid ?
                <div className="box">
                    <h2>Nouveau message</h2>
                    <AddMessage addMessage={addMessage}/>
                </div>
                : ""}

            <div className="box">
                <h1>Derniers messages publiés</h1>
                <article id="feed">
                    <MessageList messages={messages} removeMessage={removeMessage} currentUser={props.currentUser}
                                 toUserPage={props.toUserPage}/>
                </article>
            </div>
        </main>
    );

}

export default User;