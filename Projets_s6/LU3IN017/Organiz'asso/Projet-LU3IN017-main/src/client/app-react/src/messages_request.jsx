import axios from "axios";
import server_request from './server_request.jsx';


export const getMessagesRequest = async (setMessages, uid="", admin="", search="", date_from="", date_to="") => {
    try {

        let path = '/api/message/?';
        if (uid !== "")
            path += `uid=${uid}&`
        if (admin !== "")
            path += `admin=${admin}&`;
        if (search !== "")
            path += `search=${search}&`
        if (date_from !== "")
            path += `sdate=${date_from}&`
        if (date_to !== "")
            path += `edate=${date_to}`

        const response = await server_request(axios.get, path);

        setMessages(response.data);
    }
    catch(err) {
        console.error(err);
    }
}

export const addMessageRequest = (event, props, messages, setMessages, admin) => {

    event.preventDefault();
    let title = document.getElementById("message-title");
    let content = document.getElementById("message-content");
    if (title.value.length > 0 && content.value.length > 0) {
        server_request(axios.put, '/api/message', {
            title: title.value,
            content: content.value,
            uid: props.currentUser.uid,
            admin: admin
        }).then((result) => {
            let mid = result.data.details.mid;
            return server_request(axios.get, `/api/message/${mid}`);
        }).then((response) => {
            let message = response.data;
            setMessages([message, ...messages]);
        }).catch((err) => {
            console.error(err);
        });
        title.value = content.value = "";
    }
};

export const removeMessageRequest = (messages, setMessages, mid) => {
    server_request(axios.delete, `/api/message/${mid}`)
        .then(() => {
            setMessages(messages.filter(message => message.mid !== mid));
        })
        .catch((err) => {
            console.error(err);
        });
};
