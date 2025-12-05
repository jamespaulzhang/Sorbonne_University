import { useRef } from "react";

function ValidateUser(props) {
    const domModal = useRef(null);

    const modal_open = () => {
        domModal.current.style.display = "block";
    };

    const modal_close = () => {
        domModal.current.style.display = "none";
    };

    const handleAction = async (action) => {
        try {
            await action(props.user);
            modal_close();
        } catch (error) {
            console.error("Action failed:", error);
        }
    };

    return (
        <>
            <a onClick={modal_open}>{props.user.fname} {props.user.lname}</a>

            <div className="modal" ref={domModal}>
                <div className="modal-content">
                    <span className="modal-close" onClick={modal_close}>&times;</span>
                    <h3>{props.user.fname} {props.user.lname} ({props.user.uname})</h3>
                    <p>Souhaitez-vous autoriser <b>{props.user.fname}</b> à accéder au forum ?</p>
                    <div id="validation-user">
                        <button 
                            type="button" 
                            id="validation-user-btn-accept" 
                            onClick={() => handleAction(props.accept)}
                        >
                            Accepter
                        </button>
                        <button 
                            type="button" 
                            id="validation-user-btn-reject" 
                            onClick={() => handleAction(props.reject)}
                        >
                            Refuser
                        </button>
                    </div>
                </div>
            </div>
        </>
    );
}

export default ValidateUser;