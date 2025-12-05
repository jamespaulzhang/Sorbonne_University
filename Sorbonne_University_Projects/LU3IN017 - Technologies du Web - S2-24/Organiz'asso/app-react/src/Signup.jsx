import { useState } from "react";
import axios from 'axios';
import serverRequest from './api/serverConfig.jsx';

function SignUp(props) {
    const [firstName, setFirstName] = useState("");
    const [lastName, setLastName] = useState("");
    const [username, setUsername] = useState("");
    const [password1, setPassword1] = useState("");
    const [password2, setPassword2] = useState("");
    const [alert, setAlert] = useState(null);
    const [showPassword1, setShowPassword1] = useState(false);
    const [showPassword2, setShowPassword2] = useState(false);

    const getFirstName = (event) => setFirstName(event.target.value);
    const getLastName = (event) => setLastName(event.target.value);
    const getUsername = (event) => setUsername(event.target.value);
    const getPassword1 = (event) => setPassword1(event.target.value);
    const getPassword2 = (event) => setPassword2(event.target.value);

    const togglePasswordVisibility1 = () => setShowPassword1(!showPassword1);
    const togglePasswordVisibility2 = () => setShowPassword2(!showPassword2);

    const submissionHandler = () => {
        if (firstName && lastName && username && password1 && password2) {
            if (password1 === password2) {
                if (username.match(/^[A-Za-z0-9]+$/)) {
                    serverRequest(axios.put, '/api/user', {
                        fname: firstName,
                        lname: lastName,
                        uname: username,
                        password: password1
                    }).then(() => {
                        setAlert({ message: "Votre demande d'inscription a bien été transmise. Veuillez patienter le temps qu'un administrateur traite votre demande.", class: "success" });
                    }).catch((err) => {
                        console.error(err);
                        setAlert({ message: err.response.data.message, class: "error" });
                    });
                } else {
                    setAlert({ message: "Veuillez choisir un nom d'utilisateur valide.", class: "error" });
                }
            } else {
                setAlert({ message: "Les deux mots de passe saisis doivent être identiques.", class: "error" });
            }
        } else {
            setAlert({ message: "Veuillez remplir tous les champs.", class: "error" });
        }
    };

    return (
        <main className="signup-container">
            <div className="box">
                <h1>Créer un compte</h1>
                {alert && <div className={`alert ${alert.class}`}>{alert.message}</div>}
                <form id="signup">
                    <div className="form-group">
                        <div>
                            <label htmlFor="signup_firstname">Prénom</label>
                            <input type="text" id="signup_firstname" placeholder="Prénom" onChange={getFirstName} required />
                        </div>
                        <div>
                            <label htmlFor="signup_lastname">Nom</label>
                            <input type="text" id="signup_lastname" placeholder="Nom" onChange={getLastName} required />
                        </div>
                        <div>
                            <label htmlFor="signup_username">Identifiant</label>
                            <input type="text" id="signup_username" placeholder="Identifiant" onChange={getUsername} required />
                        </div>
                        <div>
                            <label htmlFor="signup_password">Mot de passe</label>
                            <div className="password-input-container">
                                <input
                                    type={showPassword1 ? "text" : "password"}
                                    id="signup_password"
                                    placeholder="Mot de passe"
                                    onChange={getPassword1}
                                    required
                                />
                                <button type="button" onClick={togglePasswordVisibility1}>
                                    {showPassword1 ? "🙈" : "👀"}
                                </button>
                            </div>
                        </div>
                        <div>
                            <label htmlFor="signup_password-confirm">Confirmer le mot de passe</label>
                            <div className="password-input-container">
                                <input
                                    type={showPassword2 ? "text" : "password"}
                                    id="signup_password-confirm"
                                    placeholder="Confirmer le mot de passe"
                                    onChange={getPassword2}
                                    required
                                />
                                <button type="button" onClick={togglePasswordVisibility2}>
                                    {showPassword2 ? "🙈" : "👀"}
                                </button>
                            </div>
                        </div>
                    </div>
                    <button type="button" onClick={submissionHandler}>S'inscrire</button>
                </form>
            </div>
        </main>
    );
}

export default SignUp;
