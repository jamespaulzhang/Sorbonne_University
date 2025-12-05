function AddMessage (props) {
  return (
      <form id="new-message">
          <div>
              <label htmlFor="message-title">Titre du message</label>
              <input type="text" id="message-title" placeholder="Veuillez saisir le titre de votre message"></input>
          </div>
          <div>
              <label htmlFor="message-content">Contenu du message</label>
              <textarea id="message-content" placeholder="Veuillez saisir votre message"></textarea>
          </div>
          <button type="button" onClick={props.addMessage}>Envoyer</button>
      </form>
  );
}

export default AddMessage;