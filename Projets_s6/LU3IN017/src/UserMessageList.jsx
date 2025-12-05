import MessageList from "./MessageList.jsx";

function UserMessageList({ messages, removeMessage, currentUser, toUserPage }) {
  return (
    <div className="box">
      <h1>Derniers messages publiés</h1>
      <article id="feed">
        <MessageList
          messages={messages}
          removeMessage={removeMessage}
          currentUser={currentUser}
          toUserPage={toUserPage}
        />
      </article>
    </div>
  );
}

export default UserMessageList;
