import { useState, useEffect } from "react";
import axios from "axios";
import serverConfig from "./api/serverConfig.jsx";
import {
  getMessagesRequest,
  addMessageRequest,
  removeMessageRequest,
} from "./api/messagesAPI.jsx";
import UserInfo from "./UserInfo.jsx";
import UserMessageList from "./UserMessageList.jsx";
import UserProfile from "./UserProfile.jsx";

function User(props) {
  const user = props.user;
  const currentUser = props.currentUser;

  const [messages, setMessages] = useState([]);
  const [isAdmin, setIsAdmin] = useState(user.admin);

  useEffect(() => {
    getMessagesRequest(
      setMessages,
      user.uid,
      currentUser.admin ? "" : "0"
    );
  }, []);

  const addMessage = (event) => {
    addMessageRequest(event, props, messages, setMessages, false);
  };

  const removeMessage = (mid) => {
    removeMessageRequest(messages, setMessages, mid);
  };

  const setAdmin = async () => {
    if (user.uid !== currentUser.uid) {
      const newStatus = !isAdmin;
      try {
        await serverConfig(axios.patch, "/api/admin", {
          uid: user.uid,
          status: newStatus,
        });
        setIsAdmin(newStatus);
      } catch (err) {
        console.error("Error updating admin status:", err);
      }
    }
  };

  return (
    <main>
      <div id="back-to-index">
        <a onClick={props.toFeedPage}>❮ Retour à l'accueil</a>
      </div>

      <UserInfo
        user={user}
        currentUser={currentUser}
        setAdmin={setAdmin}
        isAdmin={isAdmin}
      />

      <UserProfile
        user={user}
        currentUser={currentUser}
        addMessage={addMessage}
      />

      <UserMessageList
        messages={messages}
        removeMessage={removeMessage}
        currentUser={currentUser}
        toUserPage={props.toUserPage}
      />
    </main>
  );
}

export default User;


