import Message from "./Message.jsx";


function MessageList (props) {

    if(props.messages.length === 0)
        return (<div>Aucun message publié</div>);

    return (
        <>
            <ul className="message-list">
                {props.messages.map((message, index) => (
                    <li key={`message-${message.mid}`}>
                        <Message messageInfo={message} removeMessage={props.removeMessage} currentUser={props.currentUser} toUserPage={props.toUserPage} />
                    </li>
                ))}
            </ul>
        </>
    );

}

export default MessageList;