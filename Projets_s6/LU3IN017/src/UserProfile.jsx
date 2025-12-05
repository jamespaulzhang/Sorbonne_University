import { useState } from "react";
import AddMessage from "./AddMessage.jsx";

function UserProfile({ user, currentUser, addMessage }) {
  const isCurrentUser = user.uid === currentUser.uid;
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [message, setMessage] = useState("");
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deletePassword, setDeletePassword] = useState("");

  const handlePasswordChange = async (e) => {
    e.preventDefault();

    if (newPassword !== confirmPassword) {
      setMessage("Les nouveaux mots de passe ne correspondent pas.");
      return;
    }

    try {
      const response = await fetch("http://localhost:4000/api/user/change-password", {
        method: "POST",
        credentials: 'include',
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          currentPassword,
          newPassword,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessage("Mot de passe changé avec succès.");
        setCurrentPassword("");
        setNewPassword("");
        setConfirmPassword("");
      } else {
        setMessage(data.message || "Erreur lors du changement de mot de passe.");
      }
    } catch (error) {
      console.error("Error:", error);
      setMessage("Erreur de connexion au serveur.");
    }
  };

  const handleDeleteAccount = async () => {
    try {
      const response = await fetch(`http://localhost:4000/api/user/${user.uid}`, {
        method: "DELETE",
        credentials: 'include',
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          password: deletePassword,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        window.location.href = "/logout";
      } else {
        setMessage(data.message || "Erreur lors de la suppression du compte.");
        setShowDeleteModal(false);
      }
    } catch (error) {
      console.error("Error:", error);
      setMessage("Erreur de connexion au serveur.");
      setShowDeleteModal(false);
    }
  };

  return (
    <div className="box">
      <h2>Informations du profil</h2>
      <p><strong>Nom d'utilisateur :</strong> {user.uname}</p>
      <p><strong>Nom complet :</strong> {user.fname} {user.lname}</p>
      <p><strong>ID utilisateur :</strong> {user.uid}</p>
      <p><strong>Statut :</strong> {user.admin ? "Administrateur" : "Utilisateur standard"}</p>
      <p><strong>Inscrit :</strong> {user.registered ? "Oui" : "Non"}</p>

      {isCurrentUser && (
        <>
          <hr />
          <h2>Nouveau message</h2>
          <AddMessage addMessage={addMessage} />

          <hr />
          <h2>Modifier le mot de passe</h2>
          <form onSubmit={handlePasswordChange}>
            <div style={{ position: 'relative' }}>
              <label>Mot de passe actuel :</label>
              <input
                type={showCurrentPassword ? "text" : "password"}
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                required
              />
              <button
                type="button"
                style={{
                  position: 'absolute',
                  right: '10px',
                  top: '30px',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                }}
                onClick={() => setShowCurrentPassword(!showCurrentPassword)}
              >
                {showCurrentPassword ? '🙈' : '👀'}
              </button>
            </div>
            <div style={{ position: 'relative' }}>
              <label>Nouveau mot de passe :</label>
              <input
                type={showNewPassword ? "text" : "password"}
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
              />
              <button
                type="button"
                style={{
                  position: 'absolute',
                  right: '10px',
                  top: '30px',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                }}
                onClick={() => setShowNewPassword(!showNewPassword)}
              >
                {showNewPassword ? '🙈' : '👀'}
              </button>
            </div>
            <div style={{ position: 'relative' }}>
              <label>Confirmer le nouveau mot de passe :</label>
              <input
                type={showConfirmPassword ? "text" : "password"}
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
              <button
                type="button"
                style={{
                  position: 'absolute',
                  right: '10px',
                  top: '30px',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                }}
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
              >
                {showConfirmPassword ? '🙈' : '👀'}
              </button>
            </div>
            <button type="submit" className="btn">Changer le mot de passe</button>
          </form>

          {message && <p>{message}</p>}
          <hr />
          <h2>Supprimer le compte</h2>
          <button
            className="btn btn-danger"
            onClick={() => setShowDeleteModal(true)}
            style={{ backgroundColor: '#ff4444', color: 'white' }}
          >
            Supprimer mon compte
          </button>

          {showDeleteModal && (
            <div className="modal" style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0,0,0,0.5)',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              zIndex: 1000
            }}>
              <div className="modal-content" style={{
                backgroundColor: 'white',
                padding: '20px',
                borderRadius: '8px',
                maxWidth: '500px',
                width: '100%'
              }}>
                <h3>Confirmer la suppression</h3>
                <p>Êtes-vous sûr de vouloir supprimer définitivement votre compte ? Cette action est irréversible.</p>

                <div style={{ margin: '15px 0' }}>
                  <label>Entrez votre mot de passe pour confirmer :</label>
                  <input
                    type="password"
                    value={deletePassword}
                    onChange={(e) => setDeletePassword(e.target.value)}
                    style={{ width: '100%', padding: '8px' }}
                  />
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                  <button
                    onClick={() => setShowDeleteModal(false)}
                    className="btn"
                  >
                    Annuler
                  </button>
                  <button
                    onClick={handleDeleteAccount}
                    className="btn btn-danger"
                    style={{ backgroundColor: '#ff4444', color: 'white' }}
                    disabled={!deletePassword}
                  >
                    Confirmer la suppression
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      )}
      {message && <p style={{ color: message.includes('succès') ? 'green' : 'red' }}>{message}</p>}
    </div>
  );
}

export default UserProfile;
