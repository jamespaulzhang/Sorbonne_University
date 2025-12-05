Pour créer et exporter des requêtes dans Postman pour votre API hébergée sur `http://localhost:4000`, vous devez d'abord configurer chaque requête dans Postman, puis les exporter en tant que collection. Voici comment vous pouvez configurer chaque requête pour les endpoints que vous avez mentionnés :

### Exemples de Requêtes

1. **Créer un nouvel utilisateur (PUT /user)**
   - **Méthode :** PUT
   - **URL :** `http://localhost:4000/user`
   - **Body (raw - JSON) :**
     ```json
     {
         "fname": "John",
         "lname": "Doe",
         "uname": "johndoe",
         "password": "password123"
     }
     ```

2. **Récupérer tous les utilisateurs (GET /user)**
   - **Méthode :** GET
   - **URL :** `http://localhost:4000/user`

3. **Récupérer un utilisateur par ID (GET /user/{uid})**
   - **Méthode :** GET
   - **URL :** `http://localhost:4000/user/123` (remplacez `123` par l'ID de l'utilisateur)

4. **Supprimer un utilisateur (DELETE /user/{uid})**
   - **Méthode :** DELETE
   - **URL :** `http://localhost:4000/user/123` (remplacez `123` par l'ID de l'utilisateur)

5. **Connexion d'un utilisateur (POST /authentification)**
   - **Méthode :** POST
   - **URL :** `http://localhost:4000/authentification`
   - **Body (raw - JSON) :**
     ```json
     {
         "uname": "johndoe",
         "password": "password123"
     }
     ```

6. **Déconnexion d'un utilisateur (DELETE /authentification)**
   - **Méthode :** DELETE
   - **URL :** `http://localhost:4000/authentification`

7. **Récupérer les informations de l'utilisateur connecté (GET /session)**
   - **Méthode :** GET
   - **URL :** `http://localhost:4000/session`

8. **Modifier le statut administrateur d'un utilisateur (PATCH /admin)**
   - **Méthode :** PATCH
   - **URL :** `http://localhost:4000/admin`
   - **Body (raw - JSON) :**
     ```json
     {
         "uid": 123,
         "status": true
     }
     ```

9. **Récupérer les demandes d'inscription (GET /demand)**
   - **Méthode :** GET
   - **URL :** `http://localhost:4000/demand`

10. **Approuver une demande d'inscription (PATCH /demand/{uid})**
    - **Méthode :** PATCH
    - **URL :** `http://localhost:4000/demand/123` (remplacez `123` par l'ID de l'utilisateur)

11. **Supprimer une demande d'inscription (DELETE /demand/{uid})**
    - **Méthode :** DELETE
    - **URL :** `http://localhost:4000/demand/123` (remplacez `123` par l'ID de l'utilisateur)

12. **Créer un nouveau message (PUT /message)**
    - **Méthode :** PUT
    - **URL :** `http://localhost:4000/message`
    - **Body (raw - JSON) :**
      ```json
      {
          "title": "Sample Title",
          "content": "Sample Content",
          "uid": 123,
          "admin": false
      }
      ```

13. **Récupérer les messages (GET /message)**
    - **Méthode :** GET
    - **URL :** `http://localhost:4000/message`

14. **Récupérer un message par ID (GET /message/{mid})**
    - **Méthode :** GET
    - **URL :** `http://localhost:4000/message/123` (remplacez `123` par l'ID du message)

15. **Supprimer un message (DELETE /message/{mid})**
    - **Méthode :** DELETE
    - **URL :** `http://localhost:4000/message/123` (remplacez `123` par l'ID du message)

16. **Mettre à jour le contenu d'un message (PATCH /message/{mid})**
    - **Méthode :** PATCH
    - **URL :** `http://localhost:4000/message/123` (remplacez `123` par l'ID du message)
    - **Body (raw - JSON) :**
      ```json
      {
          "content": "Updated Content"
      }
      ```

17. **Créer une nouvelle réponse (PUT /reply)**
    - **Méthode :** PUT
    - **URL :** `http://localhost:4000/reply`
    - **Body (raw - JSON) :**
      ```json
      {
          "content": "Sample Reply",
          "mid": 123,
          "uid": 456
      }
      ```

18. **Récupérer les réponses (GET /reply)**
    - **Méthode :** GET
    - **URL :** `http://localhost:4000/reply`

19. **Récupérer une réponse par ID (GET /reply/{rid})**
    - **Méthode :** GET
    - **URL :** `http://localhost:4000/reply/123` (remplacez `123` par l'ID de la réponse)

20. **Supprimer une réponse (DELETE /reply/{rid})**
    - **Méthode :** DELETE
    - **URL :** `http://localhost:4000/reply/123` (remplacez `123` par l'ID de la réponse)

21. **Mettre à jour le contenu d'une réponse (PATCH /reply/{rid})**
    - **Méthode :** PATCH
    - **URL :** `http://localhost:4000/reply/123` (remplacez `123` par l'ID de la réponse)
    - **Body (raw - JSON) :**
      ```json
      {
          "content": "Updated Reply Content"
      }
      ```

22. **Mettre à jour les likes d'un message (PATCH /likes/message)**
    - **Méthode :** PATCH
    - **URL :** `http://localhost:4000/likes/message`
    - **Body (raw - JSON) :**
      ```json
      {
          "mid": 123,
          "uid": 456
      }
      ```

23. **Vérifier si un utilisateur a liké un message (GET /likes/message)**
    - **Méthode :** GET
    - **URL :** `http://localhost:4000/likes/message?mid=123&uid=456`

24. **Mettre à jour les likes d'une réponse (PATCH /likes/reply)**
    - **Méthode :** PATCH
    - **URL :** `http://localhost:4000/likes/reply`
    - **Body (raw - JSON) :**
      ```json
      {
          "rid": 123,
          "uid": 456
      }
      ```

25. **Vérifier si un utilisateur a liké une réponse (GET /likes/reply)**
    - **Méthode :** GET
    - **URL :** `http://localhost:4000/likes/reply?rid=123&uid=456`

### Exportation des Requêtes

1. **Créer une Collection :**
   - Dans Postman, cliquez sur "New" puis "Collection". Donnez un nom à votre collection et enregistrez-la.

2. **Ajouter des Requêtes à la Collection :**
   - Pour chaque requête configurée, cliquez sur "Save" et sélectionnez la collection que vous avez créée.

3. **Exporter la Collection :**
   - Cliquez sur les trois points (`...`) à côté du nom de la collection.
   - Sélectionnez "Export".
   - Choisissez le format d'export (Collection v2.1 est recommandé).
   - Cliquez sur "Export" pour télécharger le fichier JSON.

En suivant ces étapes, vous pourrez configurer et exporter toutes les requêtes nécessaires pour tester votre API dans Postman.