const express = require('express');

const Users = require('./entities/users.js');
const Messages = require('./entities/messages.js');
const Replies = require('./entities/replies.js');


/**
 * Codes de statut HTTP :
 * - 200: OK
 * - 201: OK (pour création de ressource = inscription)
 * - 400: informations incomplètes
 * - 401: accès refusé car informations incorrectes
 * - 403: accès refusé car non autorisé
 * - 404: ressource introuvable
 * - 409: conflit (= nom d'utilisateur déjà pris)
 * - 500: autre erreur
 */


/**
 * Analyse une chaîne de caractères représentant une date au format "YYYY-MM-DD" et renvoie un objet Date JavaScript.
 * Si la chaîne n'est pas dans le format correct ou représente une date invalide, la fonction renvoie null.
 *
 * @param {string} string - La chaîne à analyser.
 * @returns {Date|null} L'objet Date représentant la date, ou null si la chaîne ne peut pas être analysée.
 */
function parseDate (string) {
    if (string != null) {
        let elements;
        if ((elements = string.split('-')).length === 3) {
            let y = Number(elements[0]);
            if(!isNaN(y)) {
                let m = Number(elements[1]);
                if (!isNaN(m) && (m >= 1 && m <= 12)) {
                    m -= 1;
                    let d = Number(elements[2]);
                    if (!isNaN(d) && (d >= 1 && d <= 31)) {
                        return new Date(y, m, d);
                    }
                }
            }
        }
    }
    return null;
}


function init (db) {
    const router = express.Router();
    router.use(express.json());
    router.use(express.urlencoded({ extended: true }));
    router.use((req, res, next) => {
        res.header('Access-Control-Allow-Origin', 'http://localhost:5173');
        res.header('Access-Control-Allow-Methods', 'GET, PUT, POST, PATCH, DELETE, OPTIONS');
        res.header('Access-Control-Allow-Credentials', true);
        console.log("---------------------------------------");
        console.log("");
        console.log('API: method %s, path %s', req.method, req.path);
        console.log('Body', req.body);
        console.log("");
        next();
    });

    const users = new Users.default(db);
    const messages = new Messages.default(db);
    const replies = new Replies.default(db);

    router.route('/user')
        /**
         * Méthode : PUT
         * URL : /user
         * Paramètres : fname, lname, uname, password
         * Description : Créé un nouvel utilisateur
         */
        .put(async (req, res) => {
            try {
                let { fname, lname, uname, password } = req.body;

                if (fname != null && lname != null && uname != null && password != null) {
                    let user;
                    if ((user = await users.exists(uname)) == null) {
                        let uid = await users.create(fname, lname, uname, password);
                        res.status(201).json({
                            status: 201,
                            message: "Succès : L'utilisateur est créé.",
                            details: {
                                uid: uid
                            }
                        });
                    }
                    else {
                        res.status(409).json({
                            status: 409,
                            message: "Erreur : " + (user.registered ? "L'utilisateur renseigné existe déjà." : "Une demande d'inscription a déjà été soumise.")
                        });
                    }
                }
                else {
                    res.status(400).json({
                        status: 400,
                        message: "Erreur : Les informations sont incomplètes."
                    });
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        })

        /**
         * Méthode : GET
         * URL : /user
         * Description : Récupère tous les utilisateurs inscrits
         */
        .get(async (req, res) => {
            try {
                let params = {};

                let registered = Number(req.query.registered);
                if (!isNaN(registered) && (registered === 0 || registered === 1)) {
                    params.registered = {$eq: (registered === 1)}
                }

                let results = await users.getMany(params);
                res.send(results);
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        });

    router.route('/user/:uid')
        /**
         * Méthode : GET
         * URL : /user/{uid}
         * Description : Récupère un utilisateur inscrit à partir de son id
         */
        .get(async (req, res) => {
            try {
                let uid = req.params.uid;

                let user;
                if ((user = await users.get(uid)) != null) {
                    res.send(user);
                }
                else {
                    res.sendStatus(404);
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        })

        /**
     * Méthode : DELETE
     * URL : /user/{uid}
     * Description : Supprime un utilisateur inscrit à partir de son id
     */
    .delete(async (req, res) => {
        try {
            let uid = req.params.uid;
            const { password } = req.body;

            const user = await users.get(uid);

            if (!user) {
                return res.status(404).json({
                    status: 404,
                    message: "Erreur : L'utilisateur renseigné est inexistant."
                });
            }

            if (!req.session.admin && req.session.uid !== user.uid) {
                return res.status(403).json({
                    status: 403,
                    message: "Erreur : Vous n'avez pas la permission de supprimer cet utilisateur."
                });
            }

            if (user.password !== password) {
                return res.status(401).json({
                    status: 401,
                    message: "Erreur : Le mot de passe est incorrect."
                });
            }

            await users.delete(uid);
            res.status(200).json({
                status: 200,
                message: "Succès : L'utilisateur a été supprimé."
            });
        } catch (err) {
            res.status(500).json({
                status: 500,
                message: err.message
            });
        }
    });

    router.route('/authentification')

        /**
         * Méthode : POST
         * URL : /authentification
         * Paramètres : uname, password
         * Description : Réalise la connexion d'un utilisateur
         */
        .post(async (req, res) => {
            try {
                let { uname, password } = req.body;
                if (uname != null && password != null) {
                    let user;
                    if ((user = await users.exists(uname))) {
                        if (user.password === password) {
                            if (user.registered) {
                                req.session.regenerate((err) => {
                                    if (err) {
                                        throw err;
                                    }
                                    req.session.uid = user.uid;

                                    const minutes = 60;
                                    req.session.cookie.maxAge = minutes * 60 * 1000;

                                    res.status(200).json({
                                        status: 200,
                                        message: "",
                                        details: {
                                            user: user
                                        }
                                    });
                                });
                            }
                            else {
                                req.session.destroy((err) => {});
                                res.status(403).json({
                                    status: 403,
                                    message: "Votre demande d'inscription est en attente de validation. Veuillez patienter le temps qu'un administrateur traite votre demande."
                                });
                            }
                        }
                        else {
                            req.session.destroy((err) => {});
                            res.status(403).json({
                                status: 403,
                                message: "Le nom d'utilisateur ou le mot de passe est incorrect."
                            });
                        }
                    }
                    else {
                        res.status(401).json({
                            status: 401,
                            message: "Le nom d'utilisateur ou le mot de passe est incorrect."
                        });
                    }
                }
                else {
                    res.status(400).json({
                        status: 400,
                        message: "Les informations sont incomplètes."
                    });
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        })

        /**
         * Méthode : DELETE
         * URL : /authentification
         * Description : Réalise la déconnexion d'un utilisateur
         */
        .delete(async (req, res) => {
            try {
                req.session.destroy((err) => {});
                res.status(200).json({
                    status: 200,
                    message: "Succès : La déconnexion est validée."
                });
            }
            catch(err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        });
    
    /**
     * Méthode : POST
     * URL : /user/change-password
     * Paramètres : currentPassword, newPassword
     * Description : Permet à un utilisateur authentifié de changer son mot de passe.
     *               Vérifie si l'utilisateur est connecté, si le mot de passe actuel est correct,
     *               puis met à jour le mot de passe avec le nouveau.
     */
    router.route('/user/change-password')
        .post(async (req, res) => {
            try {
                const { currentPassword, newPassword } = req.body;
                if (!req.session.uid) {
                    return res.status(401).json({
                        status: 401,
                        message: "Non autorisé : Vous devez être connecté."
                    });
                }
                const uid = req.session.uid;
                if (currentPassword != null && newPassword != null) {
                    let user;
                    if ((user = await users.get(uid))) {
                        if (user.password === currentPassword) {
                            await users.updatePassword(uid, newPassword);
                            return res.status(200).json({
                                status: 200,
                                message: "Mot de passe changé avec succès."
                            });
                        } else {
                            return res.status(401).json({
                                status: 401,
                                message: "Le mot de passe actuel est incorrect."
                            });
                        }
                    } else {
                        return res.status(404).json({
                            status: 404,
                            message: "Utilisateur non trouvé."
                        });
                    }
                } else {
                    return res.status(400).json({
                        status: 400,
                        message: "Les informations sont incomplètes."
                    });
                }
            } catch (err) {
                console.error("Erreur changement mot de passe:", err);
                return res.status(500).json({
                    status: 500,
                    message: "Erreur interne du serveur."
                });
            }
        });
    
    /**
     * Méthode : GET
     * URL : /session
     * Description : Récupère les informations de l'utilisateur connecté
     */
    router.get('/session', async (req, res) => {
        try {
            if(req.session != null && req.session.uid != null) {
                let user;
                if((user = await users.get(req.session.uid)) != null) {
                    res.send(user);
                    return;
                }
            }
            req.session.destroy((err) => {});
            res.sendStatus(404);
        }
        catch (err) {
            res.status(500).json({
                status: 500,
                message: err.message
            });
        }
    });

    /**
     * Méthode : PATCH
     * URL : /admin
     * Paramètres : uid, status
     * Description : Modifie le statut administrateur d'un utilisateur
     */
    router.patch('/admin', async (req, res) => {
        try {
            let { uid, status } = req.body;
            if (status != null) {
                if ((typeof status) !== 'boolean') {
                    status = (status === 'true');
                }
            }

            if (uid != null && status != null) {
                if (await users.get(uid) != null) {
                    await users.setAdmin(uid, status);
                    res.status(200).json({
                        status: 200,
                        message: "Succès : " + (status ? "L'utilisateur est administrateur." : "L'utilisateur n'est plus administrateur.")
                    });
                }
                else {
                    res.status(401).json({
                        status: 401,
                        message: "Erreur : L'utilisateur renseigné est inexistant."
                    });
                }
            }
            else {
                res.status(400).json({
                    status: 400,
                    message: "Erreur : Les informations sont incomplètes."
                });
            }
        }
        catch (err) {
            res.status(500).json({
                status: 500,
                message: err.message
            });
        }
    });

    /**
     * Méthode : GET
     * URL : /demand
     * Description : Récupère les demandes d'inscription
     */
    router.get('/demand', async (req, res) => {
        try {

            let params = {};
            params.registered = false;

            let results = await users.getMany(params);
            res.send(results);
        }
        catch (err) {
            res.status(500).json({
                status: 500,
                message: err.message
            });
        }
    });

    router.route('/demand/:uid')
        /**
         * Méthode : PATCH
         * URL : /demand/{uid}
         * Paramètres : uid
         * Description : Approuve la demande d'inscription d'un utilisateur
         */
        .patch(async (req, res) => {
            try {

                let uid = req.params.uid;

                if ((await users.get(uid)) != null) {
                    await users.approve(uid);
                    res.status(200).json({
                        status: 200,
                        message: "Succès : La demande d'inscription de l'utilisateur est validée."
                    });
                }
                else {
                    res.status(401).json({
                        status: 401,
                        message: "Erreur : L'utilisateur renseigné est inexistant."
                    });
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        })

        /**
         * Méthode : DELETE
         * URL : /demand/{uid}
         * Paramètres : uid
         * Description : Supprime la demande d'inscription d'un utilisateur
         */
        .delete(async (req, res) => {
            try {

                let uid = req.params.uid;

                let user;
                if ((user = await users.get(uid)) != null) {
                    if(!user.registered) {
                        await users.decline(uid);
                        res.status(200).json({
                            status: 200,
                            message: "Succès : La demande d'inscription de l'utilisateur est rejetée."
                        });
                    }
                    else {
                        res.status(401).json({
                            status: 401,
                            message: "Erreur : L'utilisateur renseigné est déjà enregistré."
                        });
                    }
                }
                else {
                    res.status(401).json({
                        status: 401,
                        message: "Erreur : L'utilisateur renseigné est inexistant."
                    });
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        })

    router.route('/message')
        /**
         * Méthode : PUT
         * URL : /message
         * Paramètres : title, content, uid, admin
         * Description : Créé un nouveau message
         */
        .put(async (req, res) => {
            try {

                let { title, content, uid, admin } = req.body;
                if(admin != null) {
                    if((typeof admin) !== 'boolean') {
                        admin = ('true' === admin);
                    }
                }

                if (title != null && content != null && uid != null && admin != null) {
                    if (await users.get(uid) != null) {
                        let mid = await messages.create(title, content, uid, admin);
                        res.status(201).json({
                            status: 201,
                            message: "Succès : Le message est crée.",
                            details: {
                                mid: mid
                            }
                        });
                    }
                    else {
                        res.status(401).json({
                            status: 401,
                            message: "Erreur : L'utilisateur renseigné est inexistant."
                        });
                    }
                }
                else {
                    res.status(400).json({
                        status: 400,
                        message: "Erreur : Les informations sont incomplètes."
                    });
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        })

        /**
         * Méthode : GET
         * URL : /message
         * Paramètres : admin, uid, sdate, edate, search
         * Description : Récupère les messages selon plusieurs paramètres
         */
        .get(async (req, res) => {
            try {
                let params = {};

                let admin = Number(req.query.admin);
                if (!isNaN(admin) && (admin === 0 || admin === 1)) {
                    params.admin = {$eq: (admin === 1)}
                }

                if (req.query.uid != null) {
                    params.uid = {$eq: req.query.uid};
                }

                let sdate = parseDate(req.query.sdate);
                let edate = parseDate(req.query.edate);
                if (sdate != null || edate != null) {
                    params.date = {};
                    if (sdate != null) {
                        params.date['$gte'] = sdate;
                    }
                    if (edate != null) {
                        edate.setDate(edate.getDate() + 1);
                        params.date['$lte'] = edate;
                    }
                }

                let results = await messages.getMany(params);
                if (req.query.search == null) {
                    res.send(results);
                }
                else {

                    let messages = [];

                    for (const message of results) {

                        let user_message = await users.get(message.uid);

                        let replies_ = [];
                        let results_ = await replies.getMany({mid: {$eq: message.mid}});
                        for (const reply of results_) {
                            let user_reply = await users.get(reply.uid);
                            let new_reply = {...reply, ...{uname: user_reply.uname, fname: user_reply.fname, lname: user_reply.lname}};
                            replies_.push(new_reply);
                        }

                        let new_message = {...message, ...{uname: user_message.uname, fname: user_message.fname, lname: user_message.lname, replies: replies_}};
                        messages.push(new_message);
                    }

                    const search = req.query.search.toLocaleLowerCase();
                    let elements = search.split(' ');

                    let filteredResults = messages.filter((message) => {

                        let search_params = [message.title, message.content, message.uname, message.fname, message.lname];
                        let search_params_bool;

                        search_params.forEach((param) => {
                            search_params_bool = search_params_bool || search.includes(param.toLocaleLowerCase()) || param.toLocaleLowerCase().includes(search);
                            elements.forEach((element) => {
                                search_params_bool = search_params_bool || param.toLocaleLowerCase().includes(element);
                            });
                        });

                        message.replies.forEach((reply) => {
                            search_params = [reply.content, reply.uname, reply.fname, reply.lname];
                            search_params.forEach((param) => {
                                search_params_bool = search_params_bool || search.includes(param.toLocaleLowerCase()) || param.toLocaleLowerCase().includes(search);
                                elements.forEach((element) => {
                                    search_params_bool = search_params_bool || param.toLocaleLowerCase().includes(element);
                                });
                            });
                        });

                        return search_params_bool;

                    });
                    res.send(filteredResults);
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        });

    router.route('/message/:mid')
        /**
         * Méthode : GET
         * URL : /message/{mid}
         * Paramètres : mid
         * Description : Récupère un message à partir de son id
         */
        .get(async (req, res) => {
            try {
                let mid = req.params.mid;

                let message;
                if ((message = await messages.get(mid)) != null) {
                    res.send(message);
                }
                else {
                    res.sendStatus(404);
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        })

        /**
         * Méthode : DELETE
         * URL : /message
         * Paramètres : mid
         * Description : Supprime un message à partir de son id
         */
        .delete(async (req, res) => {
            try {
                let mid = req.params.mid;

                if (await messages.get(mid) != null) {
                    await messages.delete(mid);
                    res.status(200).json({
                        status: 200,
                        message: "Succès : Le message est supprimé."
                    });
                }
                else {
                    res.status(401).json({
                        status: 401,
                        message: "Erreur : Le message renseigné est inexistant."
                    });
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        })

        /**
         * Méthode : PATCH
         * URL : /message/{mid}
         * Paramètres : content
         * Description : Met à jour le contenu d'un message
         */
        .patch(async (req, res) => {
            try {
                let mid = req.params.mid;
                let { content } = req.body;

                if (content != null) {
                    if (await messages.get(mid) != null) {
                        await messages.updateContent(mid, content);
                        res.status(200).json({
                            status: 200,
                            message: "Succès : Le contenu du message a été mis à jour."
                        });
                    } else {
                        res.status(404).json({
                            status: 404,
                            message: "Erreur : Le message renseigné est inexistant."
                        });
                    }
                } else {
                    res.status(400).json({
                        status: 400,
                        message: "Erreur : Les informations sont incomplètes."
                    });
                }
            } catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        });


    router.route('/reply')
        /**
         * Méthode : PUT
         * URL : /reply
         * Paramètres : content, mid, uid
         * Description : Créé une nouvelle réponse
         */
        .put(async (req, res) => {
            try {

                let { content, mid, uid, replyTo} = req.body;

                if (content != null && mid != null && uid != null) {
                    if (await messages.get(mid) != null) {
                        if (await users.get(uid) != null) {
                            let rid = await replies.create(content, mid, uid, replyTo);
                            res.status(201).json({
                                status: 201,
                                messages: "Succès : La réponse est créée.",
                                details: {
                                    rid: rid
                                }
                            });
                        }
                        else {
                            res.status(401).json({
                                status: 401,
                                message: "Erreur : L'utilisateur renseigné est inexistant."
                            });
                        }
                    }
                    else {
                        res.status(401).json({
                            status: 401,
                            message: "Erreur : Le message renseigné est inexistant."
                        });
                    }
                }
                else {
                    res.status(400).json({
                        status: 400,
                        message: "Erreur : Les informations sont incomplètes."
                    });
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        })

        /**
         * Méthode : GET
         * URL : /reply
         * Paramètres : mid, uid
         * Description : Récupère les réponses selon plusieurs paramètres
         */
        .get(async (req, res) => {
            try {

                let params = {};

                if (req.query.mid != null) {
                    params.mid = {$eq: req.query.mid};
                }

                if (req.query.uid != null) {
                    params.uid = {$eq: req.query.uid};
                }

                let results = await replies.getMany(params);
                res.send(results);
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        });

    router.route('/reply/:rid')
        /**
         * Méthode : GET
         * URL : /reply/{rid}
         * Paramètres : rid
         * Description : Récupère une réponse à partir de son id
         */
        .get(async (req, res) => {
            try {

                let rid = req.params.rid;

                let reply;
                if ((reply = await replies.get(rid)) != null) {
                    res.send(reply);
                }
                else {
                    res.sendStatus(404);
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        })

        /**
         * Méthode : DELETE
         * URL : /reply/{rid}
         * Paramètres : rid
         * Description : Supprime une réponse à partir de son id
         */
        .delete(async (req, res) => {
            try {
                let rid = req.params.rid;
                const reply = await replies.get(rid);
                
                if (!reply) {
                    return res.status(404).json({
                        status: 404,
                        message: "Erreur : La réponse renseignée est inexistante."
                    });
                }
        
                // Get the current user from session
                const currentUser = await users.get(req.session.uid);
                
                // Check if current user is admin or the author of the reply
                if (!currentUser || (!currentUser.admin && req.session.uid !== reply.uid)) {
                    return res.status(403).json({
                        status: 403,
                        message: "Erreur : Vous n'avez pas la permission de supprimer cette réponse."
                    });
                }
        
                await replies.delete(rid);
                res.status(200).json({
                    status: 200,
                    message: "Succès : La réponse est supprimée."
                });
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        })

        /**
         * Méthode : PATCH
         * URL : /reply/{rid}
         * Paramètres : content
         * Description : Met à jour le contenu d'une réponse
         */
        .patch(async (req, res) => {
            try {
                let rid = req.params.rid;
                let { content } = req.body;
                const reply = await replies.get(rid);
                
                if (!content) {
                    return res.status(400).json({
                        status: 400,
                        message: "Erreur : Le contenu est requis."
                    });
                }

                if (!reply) {
                    return res.status(404).json({
                        status: 404,
                        message: "Erreur : La réponse renseignée est inexistante."
                    });
                }

                if (!req.session.admin && req.session.uid !== reply.uid) {
                    return res.status(403).json({
                        status: 403,
                        message: "Erreur : Vous n'avez pas la permission de modifier cette réponse."
                    });
                }

                await replies.updateContent(rid, content);
                res.status(200).json({
                    status: 200,
                    message: "Succès : Le contenu de la réponse a été mis à jour."
                });
            } catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        });

    router.route('/likes/message')
        /**
         * Méthode : PATCH
         * URL : /likes/message
         * Paramètres : mid, uid
         * Description : Met à jour les likes d'un message
         */
        .patch(async (req, res) => {
            try {

                let { mid, uid } = req.body;

                if (mid != null && uid != null) {
                    if (await messages.get(mid) != null) {
                        if (await users.get(uid) != null) {
                            let count = await messages.like(mid, uid, users);
                            res.status(200).json({
                                status: 200,
                                message: "Succès : Les likes du message ont été mis à jour.",
                                details: {
                                    count: count
                                }
                            });
                        }
                        else {
                            res.status(401).json({
                                status: 401,
                                message: "Erreur : L'utilisateur renseigné est inexistant."
                            });
                        }
                    }
                    else {
                        res.status(401).json({
                            status: 401,
                            message: "Erreur : Le message renseigné est inexistant."
                        });
                    }
                }
                else {
                    res.status(400).json({
                        status: 400,
                        message: "Erreur : Les informations sont incomplètes."
                    });
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        })

        /**
         * Méthode : GET
         * URL : /likes/message
         * Paramètres : mid, uid
         * Description : Vérifie si l'utilisateur uid a liké le message d'id mid
         */
        .get(async (req, res) => {
            try {

                let { mid, uid } = req.query;

                if (mid != null && uid != null) {
                    if (await messages.get(mid) != null) {
                        if (await users.get(uid) != null) {
                            let hasLiked = await messages.hasLiked(mid, uid);
                            res.send(hasLiked);
                        }
                        else {
                            res.status(401).json({
                                status: 401,
                                message: "Erreur : L'utilisateur renseigné est inexistant."
                            });
                        }
                    }
                    else {
                        res.status(401).json({
                            status: 401,
                            message: "Erreur : Le message renseigné est inexistant."
                        });
                    }
                }
                else {
                    res.status(400).json({
                        status: 400,
                        message: "Erreur : Les informations sont incomplètes."
                    });
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        });

    router.route('/likes/reply')
        /**
         * Méthode : PATCH
         * URL : /likes/reply
         * Paramètres : rid, uid
         * Description : Met à jour les likes d'une réponse
         */
        .patch(async (req, res) => {
            try {

                let { rid, uid } = req.body;

                if (rid != null && uid != null) {
                    if (await replies.get(rid) != null) {
                        if (await users.get(uid) != null) {
                            let count = await replies.like(rid, uid, users);
                            res.status(200).json({
                                status: 200,
                                message: "Succès : Les likes de la réponse ont été mis à jour.",
                                details: {
                                    count: count
                                }
                            });
                        }
                        else {
                            res.status(401).json({
                                status: 401,
                                message: "Erreur : L'utilisateur renseigné est inexistant."
                            });
                        }
                    }
                    else {
                        res.status(401).json({
                            status: 401,
                            message: "Erreur : La réponse renseignée est inexistante."
                        });
                    }
                }
                else {
                    res.status(400).json({
                        status: 400,
                        message: "Erreur : Les informations sont incomplètes."
                    });
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        })

        /**
         * Méthode : GET
         * URL : /likes/reply
         * Paramètres : rid, uid
         * Description : Vérifie si l'utilisateur uid a liké la réponse d'id rid
         */
        .get(async (req, res) => {
            try {

                let { rid, uid } = req.query;

                if (rid != null && uid != null) {
                    if (await replies.get(rid) != null) {
                        if (await users.get(uid) != null) {
                            let hasLiked = await replies.hasLiked(rid, uid);
                            res.send(hasLiked);
                        }
                        else {
                            res.status(401).json({
                                status: 401,
                                message: "Erreur : L'utilisateur renseigné est inexistant."
                            });
                        }
                    }
                    else {
                        res.status(401).json({
                            status: 401,
                            message: "Erreur : La réponse renseignée est inexistante."
                        });
                    }
                }
                else {
                    res.status(400).json({
                        status: 400,
                        message: "Erreur : Les informations sont incomplètes."
                    });
                }
            }
            catch (err) {
                res.status(500).json({
                    status: 500,
                    message: err.message
                });
            }
        });
    return router;
}

exports.default = init;
