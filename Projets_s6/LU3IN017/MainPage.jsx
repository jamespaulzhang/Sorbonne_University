import { useEffect, useState } from 'react';
import axios from 'axios';
import serverConfig from './api/serverConfig.jsx';
import NavigationPanel from "./NavigationPanel.jsx";
import Signup from "./Signup.jsx";
import Feed from "./Feed.jsx";
import SearchResults from "./SearchResults.jsx";
import User from "./User.jsx";
import SearchBar from "./SearchBar.jsx";
import AdminSidebar from "./AdminSidebar.jsx";
import AdminFeed from "./AdminFeed.jsx";


function MainPage(props) {
    const [currentPage, setCurrentPage] = useState({ page: "signup_page", param: null });
    const [isConnected, setConnected] = useState(false);
    const [currentUser, setCurrentUser] = useState(null);

    const getConnected = (user) => {
        setCurrentPage({ page: "feed_page", param: null });
        setConnected(true);
        setCurrentUser(user);
    };

    useEffect(() => {
        const checkSession = async () => {
            try {
                let response = await serverConfig(axios.get, '/api/session');
                let user = response.data;
                getConnected(user);
            } catch (err) {
                console.log("Session not detected...");
            }
        };
        checkSession();
    }, []);

    const setLogout = () => {
        setCurrentPage({ page: "signup_page", param: null });
        setConnected(false);
        setCurrentUser(null);
    };

    const toFeedPage = () => {
        setCurrentPage({ page: "feed_page", param: null });
    };

    const toFeedAdminPage = () => {
        if (currentUser && currentUser.admin) {
            setCurrentPage({ page: "feed_admin_page", param: null });
        }
    };

    const toResultPage = (search, date_from, date_to) => {
        setCurrentPage({
            page: "result_page",
            param: {
                search: search,
                date_from: date_from,
                date_to: date_to
            }
        });
    };

    const toUserPage = (user) => {
        setCurrentPage({ page: "user_page", param: { user } });
    };

    return (
        <div className="main-container">
            <header className="main-header">
                <div id="logo">
                    <span className="forum-title">Organiz-Asso</span>
                </div>
                <div className="header-right">
                    {isConnected && <SearchBar toResultPage={toResultPage} />}
                    <nav id="nav-login">
                        <NavigationPanel
                            login={getConnected}
                            logout={setLogout}
                            isConnected={isConnected}
                            currentUser={currentUser}
                            toUserPage={toUserPage}
                        />
                    </nav>
                </div>
            </header>
            <section className="main-section">
                {currentUser && currentUser.admin && (currentPage.page === "feed_page" || currentPage.page === "feed_admin_page") && (
                    <AdminSidebar currentPage={currentPage} toFeedPage={toFeedPage} toFeedAdminPage={toFeedAdminPage} />
                )}
                <main className="main-content">
                    {currentPage.page === "signup_page" && <Signup login={getConnected} />}
                    {currentPage.page === "feed_page" && <Feed currentUser={currentUser} toUserPage={toUserPage} />}
                    {currentPage.page === "feed_admin_page" && <AdminFeed currentUser={currentUser} toUserPage={toUserPage} />}
                    {currentPage.page === "result_page" && (
                        <SearchResults param={currentPage.param} currentUser={currentUser} toFeedPage={toFeedPage} />
                    )}
                    {currentPage.page === "user_page" && (
                        <User user={currentPage.param.user} currentUser={currentUser} toUserPage={toUserPage} toFeedPage={toFeedPage} />
                    )}
                </main>
            </section>
            <footer className="main-footer">
                <div>Yuxiang ZHANG / Antoine LECOMTE<br />LU3IN017</div>
            </footer>
        </div>
    );
}

export default MainPage;
