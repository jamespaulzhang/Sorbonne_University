import {useEffect, useState} from 'react';
import axios from 'axios';
import server_request from './server_request.jsx';
import NavigationPanel from "./components/NavigationPanel.jsx";
import Signin from "./pages/Signin.jsx";
import Feed from "./pages/Feed.jsx";
import Result from "./pages/Result.jsx";
import User from "./pages/User.jsx";
import SearchBar from "./components/SearchBar.jsx";
import AsideAdmin from "./components/AsideAdmin.jsx";
import FeedAdmin from "./pages/FeedAdmin.jsx";


function MainPage (props) {

    const [currentPage, setCurrentPage] = useState({page:"signin_page", param:null});
    const [isConnected, setConnected] = useState(false);
    const [currentUser, setCurrentUser] = useState([]);

    function getConnected (user) {
        setCurrentPage({page:"feed_page", param:null});
        setConnected(true);
        setCurrentUser(user);
    }

    useEffect(() => {
        const checkSession = async () => {
            try {
                let response = await server_request(axios.get, '/api/session');

                let user = response.data;
                //console.log(`>> Session détectée: vous êtes ${user.uname} et allez être redirigé vers la page d'accueil.`);
                //console.log(`>> ID MongoDB: ${user._id}`);
                getConnected(user);
            }
            catch (err) {
                //console.log(`>> Session non détectée...`);
            }
        }
        checkSession();
    }, []);

    function setLogout () {
        setCurrentPage({page:"signin_page", param:null});
        setConnected(false);
        setCurrentUser(null);
    }

    function toFeedPage() {
        setCurrentPage({page:"feed_page", param:null});
    }

    function toFeedAdminPage() {
        if(currentUser.admin)
            setCurrentPage({page:"feed_admin_page", param:null});
    }

    function toResultPage(search, date_from, date_to) {
        setCurrentPage({page:"result_page", param:
            {
                search: search,
                date_from: date_from,
                date_to : date_to
            }
        });
    }

    function toUserPage(user) {
        setCurrentPage({page:"user_page", param:{user}});
    }

    return (
        <>
            <header>
                <div id="logo">
                    {isConnected && currentPage.page !== "feed_page" ? <a onClick={toFeedPage}>Forum</a> : "Forum"}
                </div>
                {isConnected ? <SearchBar toResultPage={toResultPage}/> : ""}
                <nav id="nav-login">
                    <NavigationPanel login={getConnected} logout={setLogout} isConnected={isConnected} currentUser={currentUser} toUserPage={toUserPage}/>
                </nav>
            </header>
            <section>
                {(currentUser && currentUser.admin && (currentPage.page === "feed_page" || currentPage.page === "feed_admin_page")) ?
                    <AsideAdmin currentPage={currentPage} toFeedPage={toFeedPage} toFeedAdminPage={toFeedAdminPage}/> : ""}
                <main>
                    {currentPage.page === "signin_page" ? <Signin login={getConnected}/> : ""}
                    {currentPage.page === "feed_page" ? <Feed currentUser={currentUser} toUserPage={toUserPage}/> : ""}
                    {currentPage.page === "feed_admin_page" ? <FeedAdmin currentUser={currentUser} toUserPage={toUserPage}/> : ""}
                    {currentPage.page === "result_page" ?
                        <Result param={currentPage.param} currentUser={currentUser} toFeedPage={toFeedPage}/> : ""}
                    {currentPage.page === "user_page" ?
                        <User user={currentPage.param} currentUser={currentUser} toUserPage={toUserPage}
                              toFeedPage={toFeedPage}/> : ""}
                </main>
            </section>
            <footer>
                <div>Ilyas ALAHYAN / Yann ARNOULD<br/>LU3IN017</div>
            </footer>
        </>
    );

}

export default MainPage;