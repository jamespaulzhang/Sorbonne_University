import Reply from "./Reply.jsx";


function ReplyList (props) {

    return (
        <>
            <ul className="reply-list">
                {props.replies.map((reply, index) => (
                    <li key={index}>
                        <Reply id={index} replyInfo={reply} removeReply={props.removeReply} currentUser={props.currentUser} toUserPage={props.toUserPage}/>
                    </li>
                ))}
            </ul>
        </>
    );

}

export default ReplyList;