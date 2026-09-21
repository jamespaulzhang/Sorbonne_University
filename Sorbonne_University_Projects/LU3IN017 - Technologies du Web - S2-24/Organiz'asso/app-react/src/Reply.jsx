import {useState, useEffect} from "react";
import axios from 'axios';
import serverConfig from './api/serverConfig.jsx';
import EditReply from './EditReply.jsx';

function Reply (props) {
    const [author, setAuthor] = useState(null);
    const [likeCount, setLikeCount] = useState(props.replyInfo.likes.length);
    const [userHasLiked, setUserHasLiked] = useState(props.replyInfo.likes.includes(props.currentUser.uid));
    const [isEditing, setIsEditing] = useState(false);
    const [editedContent, setEditedContent] = useState(props.replyInfo.content);
    const [editedTime, setEditedTime] = useState(props.replyInfo.editedAt ? new Date(props.replyInfo.editedAt) : null);

    useEffect(() => {
        const getAuthor = async () => {
            try {
                let response = await serverConfig(axios.get, `/api/user/${props.replyInfo.uid}`);
                setAuthor(response.data);
            }
            catch(err) {
                console.error(err);
            }
        }
        getAuthor();
    }, [props.replyInfo.uid]);

    const like = () => {
        serverConfig(axios.patch, '/api/likes/reply', {
            rid: props.replyInfo.rid,
            uid: props.currentUser.uid
        })
            .then((result) => {
                setLikeCount(result.data.details.count);
                setUserHasLiked(!userHasLiked);
            })
            .catch((err) => {
                console.error(err);
            });
    };

    const editReply = () => {
        setIsEditing(true);
    };

    const saveEditedReply = () => {
        if (!editedContent.trim()) return;
        
        serverConfig(axios.patch, `/api/reply/${props.replyInfo.rid}`, {
            content: editedContent
        })
        .then(() => {
            const now = new Date();
            setIsEditing(false);
            props.replyInfo.content = editedContent;
            props.replyInfo.editedAt = now;
            setEditedTime(now);
        })
        .catch((err) => {
            console.error(err);
        });
    };

    const handleKeyDown = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            saveEditedReply();
        }
    };

    const formatDate = (date) => {
        if (!date) return '';
        return `${date.getDate().toString().padStart(2, '0')}/${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getFullYear()} ${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
    };

    let date = new Date(props.replyInfo.date);
    let formattedDate = formatDate(date);
    let formattedEditedTime = editedTime ? formatDate(editedTime) : null;

    return (
        <>
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
                        <button 
                            onClick={saveEditedReply}
                            className="btn btn-primary"
                        >
                            Enregistrer
                        </button>
                        <button 
                            onClick={() => setIsEditing(false)}
                            className="btn btn-secondary"
                        >
                            Annuler
                        </button>
                    </div>
                </div>
            ) : (
                <div className="reply-content">
                    <p>{props.replyInfo.content}</p>
                    {formattedEditedTime && (
                        <div className="edited-time">
                            (modifié le {formattedEditedTime})
                        </div>
                    )}
                </div>
            )}
            <div className="message-info">
                Publié le {formattedDate} par 
                <a onClick={() => props.toUserPage(author)}>
                    {author ? author.uname : ""}
                </a>
            </div>
            <div className="reply-icons">
                <div className="reply-icon">
                    <button type="button" className="btn-icon-reply" onClick={like}>
                        {userHasLiked ? (
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" className="reply-icon bi bi-heart-fill" viewBox="0 0 16 16">
                                <path fillRule="evenodd" d="M8 1.314C12.438-3.248 23.534 4.735 8 15-7.534 4.736 3.562-3.248 8 1.314"/>
                            </svg>
                        ) : (
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" className="reply-icon bi bi-heart" viewBox="0 0 16 16">
                                <path d="m8 2.748-.717-.737C5.6.281 2.514.878 1.4 3.053c-.523 1.023-.641 2.5.314 4.385.92 1.815 2.834 3.989 6.286 6.357 3.452-2.368 5.365-4.542 6.286-6.357.955-1.886.838-3.362.314-4.385C13.486.878 10.4.28 8.717 2.01zM8 15C-7.333 4.868 3.279-3.04 7.824 1.143q.09.083.176.171a3 3 0 0 1 .176-.17C12.72-3.042 23.333 4.867 8 15"/>
                            </svg>
                        )}
                    </button>
                    {likeCount}
                </div>

                <div className="reply-icon">
                    <button 
                        type="button" 
                        className="btn-icon-reply" 
                        onClick={() => props.onQuote && props.onQuote({
                            author: author,
                            content: props.replyInfo.content,
                            rid: props.replyInfo.rid
                        })}
                        aria-label="Citer cette réponse"
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="24"
                            height="24"
                            fill="currentColor"
                            className="reply-icon bi bi-chat-quote"
                            viewBox="0 0 16 16"
                        >
                            <path d="M2.678 11.894a1 1 0 0 1 .287.801 11 11 0 0 1-.398 2c1.395-.323 2.247-.697 2.634-.893a1 1 0 0 1 .71-.074A8 8 0 0 0 8 14c3.996 0 7-2.807 7-6s-3.004-6-7-6-7 2.808-7 6c0 1.468.617 2.83 1.678 3.894m-.493 3.905a22 22 0 0 1-.713.129c-.2.032-.352-.176-.273-.362a10 10 0 0 0 .244-.637l.003-.01c.248-.72.45-1.548.524-2.319C.743 11.37 0 9.76 0 8c0-3.866 3.582-7 8-7s8 3.134 8 7-3.582 7-8 7a9 9 0 0 1-2.347-.306c-.52.263-1.639.742-3.468 1.105"/>
                            <path d="M7.066 6.76A1.665 1.665 0 0 0 4 7.668a1.667 1.667 0 0 0 2.561 1.406c-.131.389-.375.804-.777 1.22a.417.417 0 1 0 .6.58c1.486-1.54 1.293-3.214.682-4.112zm4 0A1.665 1.665 0 0 0 8 7.668a1.667 1.667 0 0 0 2.561 1.406c-.131.389-.375.804-.777 1.22a.417.417 0 1 0 .6.58c1.486-1.54 1.293-3.214.682-4.112z"/>
                        </svg>
                    </button>
                </div>

                {props.replyInfo.uid === props.currentUser.uid && (
                    <EditReply onEdit={editReply} id={props.replyInfo.rid} />
                )}

                {(props.currentUser.admin || props.replyInfo.uid === props.currentUser.uid) && (
                    <div className="reply-icon">
                        <button
                            type="button"
                            className="btn-icon-reply"
                            onClick={() => {props.removeReply(props.replyInfo.rid)}}
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" className="reply-icon bi bi-trash3" viewBox="0 0 16 16">
                                <path d="M6.5 1h3a.5.5 0 0 1 .5.5v1H6v-1a.5.5 0 0 1 .5-.5M11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3A1.5 1.5 0 0 0 5 1.5v1H1.5a.5.5 0 0 0 0 1h.538l.853 10.66A2 2 0 0 0 4.885 16h6.23a2 2 0 0 0 1.994-1.84l.853-10.66h.538a.5.5 0 0 0 0-1zm1.958 1-.846 10.58a1 1 0 0 1-.997.92h-6.23a1 1 0 0 1-.997-.92L3.042 3.5zm-7.487 1a.5.5 0 0 1 .528.47l.5 8.5a.5.5 0 0 1-.998.06L5 5.03a.5.5 0 0 1 .47-.53Zm5.058 0a.5.5 0 0 1 .47.53l-.5 8.5a.5.5 0 1 1-.998-.06l.5-8.5a.5.5 0 0 1 .528-.47M8 4.5a.5.5 0 0 1 .5.5v8.5a.5.5 0 0 1-1 0V5a.5.5 0 0 1 .5-.5"/>
                            </svg>
                        </button>
                    </div>
                )}
            </div>
        </>
    );
}

export default Reply;