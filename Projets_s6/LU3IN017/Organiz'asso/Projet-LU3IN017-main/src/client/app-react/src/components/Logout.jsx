import axios from 'axios';
import server_request from "../server_request.jsx";


function Logout (props) {

    const logout = () => {
        server_request(axios.delete, '/api/authentification')
            .then(console.log)
            .catch(console.error)
            .finally(props.logout);
    }

    return (
        <button id="login-btn" onClick={logout}>Se déconnecter</button>
    );

}

export default Logout;