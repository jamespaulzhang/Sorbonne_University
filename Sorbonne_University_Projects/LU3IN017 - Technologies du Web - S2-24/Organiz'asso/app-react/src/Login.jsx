import { useState, useRef } from "react";
import axios from 'axios';
import serverConfig from "./api/serverConfig.jsx";

function Login(props) {
    const domModal = useRef(null);

    const modalOpen = () => {
        domModal.current.style.display = "block";
    };

    const modalClose = () => {
        domModal.current.style.display = "none";
    };

    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [showPassword, setShowPassword] = useState(false);

    const getUsername = (event) => {
        setUsername(event.target.value);
    };

    const getPassword = (event) => {
        setPassword(event.target.value);
    };

    const togglePasswordVisibility = () => {
        setShowPassword(!showPassword);
    };

    const [alert, setAlert] = useState(null);

    const handleLogin = () => {
        if (username && password) {
            serverConfig(axios.post, '/api/authentification', {
                uname: username,
                password: password
            }).then((result) => {
                let user = result.data.details.user;
                setAlert(null);
                props.login(user);
            }).catch((err) => {
                console.error(err);
                setAlert(err.response.data.message);
            });
        } else {
            setAlert("Les informations saisies sont incomplètes.");
        }
    };

    const handleKeyDown = (event) => {
        if (event.key === 'Enter') {
            handleLogin();
        }
    };

    return (
        <div className="login-container">
            <button id="login-btn" onClick={modalOpen}>Se connecter</button>
            <div className="modal" ref={domModal}>
                <div className="modal-content">
                    <span className="modal-close" onClick={modalClose}>&times;</span>
                    <h2>Se connecter</h2>
                    {alert && <div className='alert error'>{alert}</div>}
                    <form id="login" method="post" action="" onKeyDown={handleKeyDown}>
                        <div className="form-group">
                            <label htmlFor="username">Identifiant</label>
                            <input
                                type="text"
                                id="username"
                                placeholder="Identifiant"
                                onChange={getUsername}
                                required
                            />
                        </div>
                        <div className="form-group">
                            <label htmlFor="password">Mot de passe</label>
                            <div className="password-input-container">
                                <input
                                    type={showPassword ? "text" : "password"}
                                    id="password"
                                    placeholder="Mot de passe"
                                    onChange={getPassword}
                                    required
                                />
                                <button type="button" onClick={togglePasswordVisibility}>
                                    {showPassword ? "🙈" : "👀"}
                                </button>
                            </div>
                        </div>
                        <button type="button" onClick={handleLogin}>Connexion</button>
                    </form>
                </div>
            </div>
        </div>
    );
}

export default Login;
