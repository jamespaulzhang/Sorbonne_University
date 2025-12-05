import { useState, useEffect } from "react";
import axios from "axios";
import serverConfig from "./api/serverConfig.jsx";
import ReplyList from "./ReplyList.jsx";
import DeleteMessage from "./DeleteMessage.jsx";
import EditMessage from "./EditMessage.jsx";

function Message(props) {
  const [author, setAuthor] = useState(null);
  const [replies, setReplies] = useState([]);
  const [replyCount, setReplyCount] = useState(0);
  const [likeCount, setLikeCount] = useState(props.messageInfo.likes.length);
  const [userHasLiked, setUserHasLiked] = useState(
    props.messageInfo.likes.includes(props.currentUser.uid)
  );
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState(props.messageInfo.content);
  const [editedTime, setEditedTime] = useState(null);

  useEffect(() => {
    const getAuthor = async () => {
      try {
        let response = await serverConfig(
          axios.get,
          `/api/user/${props.messageInfo.uid}`
        );
        setAuthor(response.data);
      } catch (err) {
        console.error(err);
      }
    };
    getAuthor();
  }, [props.messageInfo.uid]);

  const getReplies = async () => {
    try {
      let response = await serverConfig(
        axios.get,
        `/api/reply?mid=${props.messageInfo.mid}`
      );
      setReplies(response.data);
      setReplyCount(response.data.length);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    getReplies();
  }, [props.messageInfo.mid]);

  const addReply = (event) => {
    event.preventDefault();
    let content = document.getElementById(`reply-${props.messageInfo.mid}`);
    if (content.value.length > 0) {
      let user = props.currentUser;
      let message = props.messageInfo;
      serverConfig(axios.put, "/api/reply", {
        content: content.value,
        uid: user.uid,
        mid: message.mid,
      })
        .then((result) => {
          let rid = result.data.details.rid;
          return serverConfig(axios.get, `/api/reply/${rid}`);
        })
        .then((response) => {
          let reply = response.data;
          setReplies([...replies, reply]);
          getReplies();
        })
        .catch((err) => {
          console.error(err);
        });
      content.value = "";
    }
  };

  const removeReply = (rid) => {
    serverConfig(axios.delete, `/api/reply/${rid}`)
        .then((result) => {
            setReplies(replies.filter((reply) => reply.rid !== rid));
            getReplies();
        })
        .catch((err) => {
            console.error(err);
        });
  };

  const like = () => {
    serverConfig(axios.patch, "/api/likes/message", {
      mid: props.messageInfo.mid,
      uid: props.currentUser.uid,
    })
      .then((result) => {
        setLikeCount(result.data.details.count);
        setUserHasLiked(!userHasLiked);
      })
      .catch((err) => {
        console.error(err);
      });
  };

  const editMessage = () => setIsEditing(true);

  const saveEditedMessage = () => {
    serverConfig(axios.patch, `/api/message/${props.messageInfo.mid}`, {
      content: editedContent,
    })
      .then(() => {
        setIsEditing(false);
        setEditedTime(new Date());
        props.messageInfo.content = editedContent;
      })
      .catch((err) => {
        console.error(err);
      });
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      saveEditedMessage();
    }
  };

  const startReplyWithQuote = () => {
    setReplyingTo(props.messageInfo);
    const textarea = document.getElementById(`reply-${props.messageInfo.mid}`);
    if (textarea) {
      textarea.value = `@${author?.uname} "${props.messageInfo.content}"\n`;
      textarea.focus();
    }
  };

  const formatDate = (date) => {
    return `${date.getDate().toString().padStart(2, "0")}/${(date.getMonth() + 1)
      .toString()
      .padStart(2, "0")}/${date.getFullYear()} ${date
      .getHours()
      .toString()
      .padStart(2, "0")}:${date.getMinutes().toString().padStart(2, "0")}`;
  };

  let date = new Date(props.messageInfo.date);
  let formattedDate = formatDate(date);
  let formattedEditedTime = editedTime ? formatDate(editedTime) : null;

  return (
    <div className="message-container">
      <h3>{props.messageInfo.title}</h3>
      {isEditing ? (
        <div className="edit-reply-container">
          <textarea
            value={editedContent}
            onChange={(e) => setEditedContent(e.target.value)}
            onKeyDown={handleKeyDown}
            autoFocus
            className="edit-reply-textarea"
          />
          <div className="edit-reply-buttons">
            <button onClick={saveEditedMessage} className="btn btn-primary">
              Enregistrer
            </button>
            <button
              onClick={() => {
                setIsEditing(false);
                setEditedContent(props.messageInfo.content);
              }}
              className="btn btn-secondary"
            >
              Annuler
            </button>
          </div>
        </div>
      ) : (
        <p>
          {props.messageInfo.content}
          {formattedEditedTime && (
            <span> (modifié le {formattedEditedTime})</span>
          )}
        </p>
      )}

      <div className="message-info">
        Publié le {formattedDate} par{" "}
        <a onClick={() => props.toUserPage(author)}>
          {author ? author.uname : ""}
        </a>
      </div>

      <div className="reply-icons">
        <div className="reply-icon">
          <button type="button" className="btn-icon-reply" onClick={like}>
            {userHasLiked ? (
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" className="reply-icon bi bi-heart-fill" viewBox="0 0 16 16">
                <path fillRule="evenodd" d="M8 1.314C12.438-3.248 23.534 4.735 8 15-7.534 4.736 3.562-3.248 8 1.314" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" className="reply-icon bi bi-heart" viewBox="0 0 16 16">
                <path d="m8 2.748-.717-.737C5.6.281 2.514.878 1.4 3.053c-.523 1.023-.641 2.5.314 4.385.92 1.815 2.834 3.989 6.286 6.357 3.452-2.368 5.365-4.542 6.286-6.357.955-1.886.838-3.362.314-4.385C13.486.878 10.4.28 8.717 2.01zM8 15C-7.333 4.868 3.279-3.04 7.824 1.143q.09.083.176.171a3 3 0 0 1 .176-.17C12.72-3.042 23.333 4.867 8 15" />
              </svg>
            )}
          </button>
          {likeCount}
        </div>

        <div className="reply-icon">
          <button
            type="button"
            className="btn-icon-reply"
            onClick={startReplyWithQuote}
            aria-label="Citer ce message"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" className="reply-icon bi bi-chat-quote" viewBox="0 0 16 16">
              <path d="M2.678 11.894a1 1 0 0 1 .287.801 11 11 0 0 1-.398 2c1.395-.323 2.247-.697 2.634-.893a1 1 0 0 1 .71-.074A8 8 0 0 0 8 14c3.996 0 7-2.807 7-6s-3.004-6-7-6-7 2.808-7 6c0 1.468.617 2.83 1.678 3.894m-.493 3.905a22 22 0 0 1-.713.129c-.2.032-.352-.176-.273-.362a10 10 0 0 0 .244-.637l.003-.01c.248-.72.45-1.548.524-2.319C.743 11.37 0 9.76 0 8c0-3.866 3.582-7 8-7s8 3.134 8 7-3.582 7-8 7a9 9 0 0 1-2.347-.306c-.52.263-1.639.742-3.468 1.105" />
            </svg>
          </button>
          {replyCount}
        </div>

        {props.messageInfo.uid === props.currentUser.uid && (
          <EditMessage onEdit={editMessage} id={props.messageInfo.mid} />
        )}
        {(props.currentUser.admin || props.messageInfo.uid === props.currentUser.uid) && (
          <DeleteMessage onDelete={props.removeMessage} id={props.messageInfo.mid} />
        )}
      </div>

      <ReplyList
        replies={replies}
        setReplies={setReplies}
        removeReply={removeReply}
        currentUser={props.currentUser}
        toUserPage={props.toUserPage}
        onQuote={startReplyWithQuote}
        replyInfo={{ mid: props.messageInfo.mid }}
      />

      <form className="new-reply">
        <div>
          <label htmlFor={`reply-${props.messageInfo.mid}`}>Votre réponse</label>
          <textarea
            className="reply-content"
            id={`reply-${props.messageInfo.mid}`}
            placeholder="Veuillez saisir votre message de réponse"
          ></textarea>
        </div>
        <button type="button" className="reply-button" onClick={addReply}>
          Ajouter une réponse
        </button>
      </form>
    </div>
  );
}

export default Message;
