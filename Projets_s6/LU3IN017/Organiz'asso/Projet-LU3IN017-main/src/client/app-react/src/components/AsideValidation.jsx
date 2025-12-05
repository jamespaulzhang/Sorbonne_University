import {useState, useEffect} from "react";
import axios from 'axios';
import server_request from "../server_request.jsx";
import AsideValidationUser from "./AsideValidationUser.jsx";


function AsideValidation (props) {

    const [users, setUsers] = useState([]);

    const getUnregisteredUsers = async () => {
        try {
            let response = await server_request(axios.get, '/api/user/?registered=0');
            setUsers(response.data);
        }
        catch(err) {
            console.error(err);
        }
    }

    useEffect(() => {
        getUnregisteredUsers();
    } , []);

    function accept_user(user) {
        server_request(axios.patch, `/api/demand/${user.uid}`)
            .then()
            .catch(console.error)
            .finally(getUnregisteredUsers);
    }

    function reject_user(user) {
        server_request(axios.delete, `/api/demand/${user.uid}`)
            .then()
            .catch(console.error)
            .finally(getUnregisteredUsers);
    }

    return <>
        <div className="box">
            <h3>En attente de validation</h3>
            <ul className="">
                { users.length === 0 ?
                    <div>Aucun</div>
                :
                    <>
                        {users.map((user, index) => (
                            <li key={index}>
                                <AsideValidationUser id={index} user={user} accept={accept_user} reject={reject_user} />
                            </li>
                        ))}
                    </>
                }
            </ul>
        </div>
    </>;

}

export default AsideValidation;