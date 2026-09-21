const Messages = require('./messages.js');

class Users {

    constructor (db) {
        this.db = db;
        this.col = this.db.collection('users');
    }

    async exists (uname) {
        return await this.col.findOne({
            uname: {$eq: uname}
        });
    }

    async create (fname, lname, uname, password) {
        if (!await this.exists(uname)) {

            let user = { fname: fname, lname: lname, uname: uname, password: password };

            user.admin = false;
            user.registered = false;

            let doc = await this.col.insertOne(user);

            let strid = doc.insertedId.toString();
            await this.col.updateOne(
                {
                    _id: {$eq: doc.insertedId}
                },
                {
                    $set: {
                        uid: strid
                    }
                }
            )
            return strid;
        }
        return null;
    }

    async get (uid) {
        return await this.col.findOne({
            uid: {$eq: uid}
        });
    }

    async getMany (params) {
        return await this.col.find(params).toArray();
    }

    async delete (uid) {
        if (await this.get(uid) != null) {
            let messages = new Messages.default(this.db);

            let relatedMessages = await messages.getMany({
                uid: {$eq: uid}
            });
            for (let i = 0; i < relatedMessages.length; i++) {
                await messages.delete(relatedMessages[i].mid);
            }

            await this.col.deleteOne({
                uid: {$eq: uid}
            });
            return true;
        }
        return false;
    }

    async setAdmin (uid, status) {
        if (await this.get(uid) != null && (typeof status) === 'boolean') {
            await this.col.updateOne(
                {
                    uid: {$eq: uid}
                },
                {
                    $set: {
                        admin: status
                    }
                }
            );
            return true;
        }
        return false;
    }

    async approve(uid) {
        if (await this.get(uid) != null) {
            await this.col.updateOne(
                {
                    uid: {$eq: uid}
                },
                {
                    $set: {
                        registered: true
                    }
                }
            );
            return true;
        }
        return false;
    }

    async decline (uid) {
        let user;
        if((user = await this.get(uid)) != null && !user.registered) {
            await this.delete(uid);
        }
    }

    async updatePassword(uid, newPassword) {
        try {
            const result = await this.col.updateOne(
                { uid: uid },
                { $set: { password: newPassword } }
            );
            
            if (result.modifiedCount === 0) {
                throw new Error("Aucun utilisateur trouvé ou mot de passe identique");
            }
            
            return true;
        } catch (err) {
            console.error("Erreur updatePassword:", err);
            throw err;
        }
    }
}

exports.default = Users;
