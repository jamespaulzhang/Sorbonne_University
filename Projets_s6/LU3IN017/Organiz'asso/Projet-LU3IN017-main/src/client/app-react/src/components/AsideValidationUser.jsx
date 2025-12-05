import {useRef, useState, useEffect} from "react";


function AsideValidationUser (props) {

    const domModal = useRef(null);

    let modal_open = function () {
        domModal.current.style.display = "block";
    };
    let modal_close = function () {
        domModal.current.style.display = "none";
    };

    function accept_user(user) {
        props.accept(user);
        modal_close();
    }

    function reject_user(user) {
        props.reject(user);
        modal_close();
    }

    return <>
        <a onClick={modal_open}>{props.user.fname} {props.user.lname}</a>

        <div className="modal" ref={domModal}>

            <div className="modal-content">

                <span className="modal-close" onClick={modal_close}>&times;</span>

                <h3>{props.user.fname} {props.user.lname} ({props.user.uname})</h3>
                <p>Souhaitez-vous autoriser <b>{props.user.fname}</b> à accéder au forum ?</p>
                <div id="validation-user">
                    <button type="button" id="validation-user-btn-accept" onClick={() => accept_user(props.user)}>Accepter</button>
                    <button type="button" id="validation-user-btn-reject" onClick={() => reject_user(props.user)}>Refuser</button>
                </div>

            </div>

        </div>

    </>;

}

export default AsideValidationUser;