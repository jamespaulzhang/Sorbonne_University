import Login from "./Login.jsx";
import Logout from "./Logout.jsx";


function NavigationPanel (props) {

    if(props.isConnected)
        return (
            <>
                <button id="login-btn" onClick={() => props.toUserPage(props.currentUser)}>Mon profil</button>
                <Logout logout={props.logout} />
            </>
        );
    else
        return (
            <Login login={props.login}/>
        );
}

export default NavigationPanel;