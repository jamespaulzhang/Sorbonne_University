import axios from 'axios';
import serverConfig from './api/serverConfig.jsx';
import Reply from "./Reply.jsx";

function ReplyList(props) {
    const removeReply = async (rid) => {
        try {
            const reply = await serverConfig(axios.get, `/api/reply/${rid}`);
            if (reply.data) {
                if (props.currentUser.admin || reply.data.uid === props.currentUser.uid) {
                    const response = await serverConfig(axios.delete, `/api/reply/${rid}`);
                    if (response.status === 200) {
                        if (props.onReplyRemoved) {
                            props.onReplyRemoved(rid);
                        }
                        console.log('Reply deleted successfully!');
                        const updatedReplies = await serverConfig(axios.get, `/api/reply?mid=${props.replyInfo.mid}`);
                        props.setReplies(updatedReplies.data);
                    }
                } else {
                    console.error('You do not have permission to delete this reply.');
                }
            }
        } catch (error) {
            console.error('Failed to delete the reply:', error);
            if (error.response) {
                console.error('Error:', error.response.data.message);
            }
        }
    };

    return (
        <>
            <ul className="reply-list">
                {props.replies.map((reply, index) => (
                    <li key={reply.rid || index}>
                        <Reply
                            id={index}
                            replyInfo={reply}
                            removeReply={removeReply}
                            currentUser={props.currentUser}
                            toUserPage={props.toUserPage}
                            onQuote={props.onQuote}
                        />
                    </li>
                ))}
            </ul>
        </>
    );
}

export default ReplyList;
