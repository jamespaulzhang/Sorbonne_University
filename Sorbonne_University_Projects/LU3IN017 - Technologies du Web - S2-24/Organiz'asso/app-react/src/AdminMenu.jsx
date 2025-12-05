import { useState } from "react";
import UserValidation from "./UserValidation";

function AdminMenu(props) {
    return (
        <div className="box">
            <h3>Forums</h3>
            <ul>
                <li>
                    {props.currentPage.page !== "feed_page" ? (
                        <a onClick={props.toFeedPage}>Forum membres</a>
                    ) : "Forum membres"}
                </li>
                <li>
                    {props.currentPage.page !== "feed_admin_page" ? (
                        <a onClick={props.toFeedAdminPage}>Forum administrateurs</a>
                    ) : "Forum administrateurs"}
                </li>
            </ul>
            
            <h3>Administration</h3>
            <ul>
                <li>
                    <div 
                        onClick={() => props.setShowValidation(!props.showValidation)}
                        style={{ cursor: 'pointer' }}
                    >
                        Validation des utilisateurs {props.showValidation ? '▼' : '▶'}
                    </div>
                </li>
            </ul>
        </div>
    );
}

export default AdminMenu;