# Projet de Gestion de Comptes Utilisateurs

Ce projet est une application web pour la gestion de comptes utilisateurs. Il permet aux utilisateurs de s'inscrire, de se connecter, de modifier leur mot de passe et de supprimer leur compte. Les administrateurs peuvent également approuver ou rejeter les demandes d'inscription et gérer le statut de membre Admin pour les utilisateurs existants.

## Fonctionnalités

- **Inscription et Connexion** : Les utilisateurs peuvent s'inscrire et se connecter à leur compte.
- **Modification du Mot de Passe** : Les utilisateurs peuvent modifier leur mot de passe.
- **Suppression de Compte** : Les utilisateurs peuvent supprimer leur compte après confirmation avec leur mot de passe.
- **Gestion des Utilisateurs** : Les administrateurs peuvent approuver ou rejeter les demandes d'inscription et gérer les utilisateurs. 
- **Forums** : Forum utilisateurs et Forum Administrateurs.
- **Messages et Réponses** : Les utilisateurs peuvent créer/modifier/supprimer leurs messages et y répondre. Les administrateurs peuvent également supprimer les messages des autres.
- **Likes** : Les utilisateurs peuvent liker les messages et les réponses.

## Prérequis

- Postman/Curl
- React
- Node.js
- MongoDB

## Démarrage du Projet

Pour démarrer le projet, suivez ces étapes :

1. Accédez au dossier `app-react` :

    ```sh
    cd app-react
    ```

2. Installez les dépendances et démarrez l'application frontend :

    ```sh
    npm install
    npm run dev
    ```

3. Démarrez MongoDB :

    Assurez-vous que MongoDB est installé et en cours d'exécution.

4. Accédez au dossier `server` :

    ```sh
    cd ../server
    ```

5. Installez les dépendances et démarrez l'application backend :

    ```sh
    npm install
    node src/index.js
    ```

6. Ouvrez votre navigateur et accédez à `http://localhost:5173`.

## Structure du Projet

- `app-react/` : Contient le code source de l'application frontend.
  - `src/` : Contient tous les composants React.
  - `style/`: Contient le fichier CSS.
- `server/` : Contient le code source de l'application backend.
  - `src/` : Contient tous les codes NodeJS.
    - `entities/` : Contient les modèles de données.

## Utilisation

### Inscription

1. Accédez à la page d'inscription.
2. Remplissez le formulaire d'inscription avec vos informations.
3. Soumettez le formulaire.

### Connexion

1. Accédez à la page de connexion.
2. Entrez votre nom d'utilisateur et votre mot de passe.
3. Soumettez le formulaire.

### Modification du Mot de Passe

1. Connectez-vous à votre compte.
2. Accédez à la page de modification du mot de passe.
3. Entrez votre mot de passe actuel et le nouveau mot de passe.
4. Soumettez le formulaire.

### Suppression de Compte

1. Connectez-vous à votre compte.
2. Accédez à la page de suppression de compte.
3. Entrez votre mot de passe pour confirmer.
4. Soumettez le formulaire.

### Gestion des Utilisateurs (Administrateur)

1. Connectez-vous en tant qu'administrateur.
2. Accédez à la page de gestion des utilisateurs.
3. Approuvez ou rejetez les demandes d'inscription.
4. Gérez le statut de membre Admin pour les utilisateurs existants.

### Messages et Réponses

1. Connectez-vous à votre compte.
2. Accédez à la page des messages.
3. Créez un nouveau message ou répondez à un message existant.

### Likes

1. Connectez-vous à votre compte.
2. Accédez à la page des messages ou des réponses.
3. Likez un message ou une réponse.