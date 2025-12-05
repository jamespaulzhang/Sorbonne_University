import axios from 'axios';
import serverConfig from "./api/serverConfig.jsx";

function Logout (props) {
    const logout = () => {
        serverConfig(axios.delete, '/api/authentification')
            .then(console.log)
            .catch(console.error)
            .finally(props.logout);
    }

    return (
        <button id="login-btn" onClick={logout}>Se déconnecter</button>
    );
}

export default Logout;