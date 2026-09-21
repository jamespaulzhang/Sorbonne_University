import { useState } from "react";
import AdminMenu from "./AdminMenu.jsx";
import UserValidation from "./UserValidation.jsx";
import axios from 'axios';
import serverConfig from "./api/serverConfig.jsx";

function AdminSidebar(props) {
    const [showValidation, setShowValidation] = useState(false);

    const handleAcceptUser = async (user) => {
        try {
            await serverConfig(axios.patch, `/api/demand/${user.uid}`);
            return true;
        } catch (error) {
            console.error("Failed to accept user:", error);
            return false;
        }
    };

    const handleRejectUser = async (user) => {
        try {
            await serverConfig(axios.delete, `/api/demand/${user.uid}`);
            return true;
        } catch (error) {
            console.error("Failed to reject user:", error);
            return false;
        }
    };

    return (
        <aside>
            <AdminMenu
                currentPage={props.currentPage}
                toFeedPage={props.toFeedPage}
                toFeedAdminPage={props.toFeedAdminPage}
                showValidation={showValidation}
                setShowValidation={setShowValidation}
            />
            {showValidation && (
                <UserValidation
                    accept_user={handleAcceptUser}
                    reject_user={handleRejectUser}
                />
            )}
        </aside>
    );
}

export default AdminSidebar;