import { useState, useEffect } from "react";
import axios from 'axios';
import serverConfig from "./api/serverConfig.jsx";
import ValidateUser from "./ValidateUser.jsx";

function UserValidation(props) {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);

    const fetchUsers = async () => {
        setLoading(true);
        try {
            const response = await serverConfig(axios.get, '/api/user/?registered=0');
            setUsers(response.data);
        } catch(err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchUsers();
    }, []);

    const handleActionComplete = async (actionFn, user) => {
        const success = await actionFn(user);
        if (success) {
            await fetchUsers();
        }
    };

    if (loading) return <div>Chargement...</div>;

    return (
        <div className="box">
            <h4>En attente de validation</h4>
            {users.length === 0 ? (
                <div>Aucun</div>
            ) : (
                <ul>
                    {users.map((user, index) => (
                        <li key={index}>
                            <ValidateUser 
                                user={user} 
                                accept={(user) => handleActionComplete(props.accept_user, user)}
                                reject={(user) => handleActionComplete(props.reject_user, user)}
                            />
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}

export default UserValidation;