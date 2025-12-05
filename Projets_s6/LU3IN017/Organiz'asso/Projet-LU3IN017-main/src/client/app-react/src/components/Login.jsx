import {useState, useRef} from "react";
import axios from 'axios';
import server_request from "../server_request.jsx";


function Login (props) {

    /* Login modal */
    const domModal = useRef(null);
    let modal_open = function () {
        domModal.current.style.display = "block";
    };
    let modal_close = function () {
        domModal.current.style.display = "none";
    };

    /* Login form */
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const getUsername = (event) => {
        setUsername(event.target.value);
    }
    const getPassword = (event) => {
        setPassword(event.target.value);
    }

    /* Message */
    const [alert, setAlert] = useState(null);

    /* Gestion de la connexion */
    const login = () => {
        if (username && password) {
            server_request(axios.post, '/api/authentification', {
                uname: username,
                password: password
            }).then((result) => {
                let user = result.data.details.user;
                setAlert(null);
                props.login(user);
            }).catch((err) => {
                console.error(err);
                setAlert(err.response.data.message);
            })
        }
        else {
            setAlert("Les informations saisies sont incomplètes.");
        }
    }

    return (
        <>
            <button id="login-btn" onClick={modal_open}>Se connecter</button>
            <div className="modal" ref={domModal}>

                <div className="modal-content">

                    <span className="modal-close" onClick={modal_close}>&times;</span>

                    <h2>Se connecter</h2>

                    { (alert != null) ? <div className='alert error'>{alert}</div> : <></> }

                    <form id="login" method="post" action="">
                        <div>
                            <label htmlFor="username">Identifiant</label>
                            <input type="text" id="username" placeholder="Identifiant" onChange={getUsername} required/>
                        </div>
                        <div>
                            <label htmlFor="password">Mot de passe</label>
                            <input type="password" id="password" placeholder="Mot de passe" onChange={getPassword} required/>
                        </div>
                        <button type="button" onClick={login}>Connexion</button>
                    </form>

                </div>

            </div>
        </>
    );

}

export default Login;