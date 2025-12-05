import {useState} from "react";
import axios from 'axios';
import server_request from '../server_request.jsx';


function Signin (props) {

    const [firstName, setFirstName] = useState("");
    const [lastName, setLastName] = useState("");
    const [username, setUsername] = useState("");
    const [password1, setPassword1] = useState("");
    const [password2, setPassword2] = useState("");

    const getFirstName = (event) => {setFirstName(event.target.value)};
    const getLastName = (event) => {setLastName(event.target.value)};
    const getUsername = (event) => {setUsername(event.target.value)};
    const getPassword1 = (event) => {setPassword1(event.target.value)};
    const getPassword2 = (event) => {setPassword2(event.target.value)};

    /* Message */
    const [alert, setAlert] = useState(null);


    /* Gestion de l'inscription */
    const submissionHandler = () => {

        if (firstName !== "" && lastName !== "" && username !== "" && password1 !== "" && password2 !== "") {

            if (password1 === password2) {

                if(username.match(/^[A-Za-z0-9]+$/)) {

                    server_request(axios.put, '/api/user', {
                        fname: firstName,
                        lname: lastName,
                        uname: username,
                        password: password1
                    }).then((result) => {
                        setAlert({message:"Votre demande d'inscription a bien été transmise. Veuillez patienter le temps qu'un administrateur traite votre demande.", class:"success"});
                    }).catch((err) => {
                        console.error(err);
                        setAlert({message:err.response.data.message, class:"error"});
                    });

                } else {
                    setAlert({message:"Veuillez choisir un nom d'utilisateur valide.", class:"error"});
                }

            } else {
                setAlert({message:"Les deux mots de passe saisis doivent être identiques.", class:"error"});
            }

        } else {
            setAlert({message:"Veuillez remplir tous les champs.", class:"error"});
        }

    }

    return (
        <main>
            <div className="box">
                <h1>Créer un compte</h1>
                {alert != null ?
                    <div className={"alert " + alert.class}>{alert.message}</div>
                : ""}
                <form id="signin">
                    <div>
                        <div>
                            <label htmlFor="signin_firstname">Prénom</label>
                            <input type="text" id="signin_firstname" placeholder="Prénom" onChange={getFirstName} required></input>
                        </div>
                        <div>
                            <label htmlFor="signin_lastname">Nom</label>
                            <input type="text" id="signin_lastname" placeholder="Nom" onChange={getLastName} required></input>
                        </div>
                        <div>
                            <label htmlFor="signin_username">Identifiant</label>
                            <input type="text" id="signin_username" placeholder="Identifiant" onChange={getUsername} required></input>
                        </div>
                        <div>
                            <label htmlFor="signin_password">Mot de passe</label>
                            <input type="password" id="signin_password" placeholder="Mot de passe" onChange={getPassword1} required></input>
                        </div>
                        <div>
                            <label htmlFor="signin_password-confirm">Confirmer le mot de passe</label>
                            <input type="password" id="signin_password-confirm" placeholder="Confirmer le mot de passe" onChange={getPassword2} required></input>
                        </div>
                    </div>

                    <button type="button" onClick={submissionHandler}>S'inscrire</button>
                </form>
            </div>
        </main>
    );

}

export default Signin;