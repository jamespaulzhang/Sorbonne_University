const Replies = require('./replies.js');

class Messages {

    constructor (db) {
        this.db = db;
        this.col = this.db.collection('messages');
    }

    async create (title, content, uid, admin) {
        let date = new Date();
        let message = { title: title, content: content, date: date, uid: uid, admin: admin, likes: [] };

        let doc = await this.col.insertOne(message);

        let strid = doc.insertedId.toString();
        await this.col.updateOne(
            {
                _id: {$eq: doc.insertedId}
            },
            {
                $set: {
                    mid: strid
                }
            }
        )
        return strid;
    }

    async get (mid) {
        return await this.col.findOne({
            mid: {$eq: mid}
        });
    }

    async getMany (params, sorting=-1) {
        return await this.col.find(params).sort({date: sorting}).toArray();
    }

    async delete (mid) {
        if(await this.get(mid) != null) {
            let replies = new Replies.default(this.db);

            let relatedReplies = await replies.getMany({
                mid: {$eq: mid}
            });
            for(let i = 0; i < relatedReplies.length; i++) {
                await replies.delete(relatedReplies[i].rid);
            }

            await this.col.deleteOne({
                mid: {$eq: mid}
            });
            return true;
        }
        return false;
    }

    async like (mid, uid, users) {
        let message;
        if((message = await this.get(mid)) != null) {
            let index;
            if((index = message.likes.indexOf(uid)) > -1) {
                message.likes.splice(index, 1);
            }
            else {
                if(await users.get(uid) == null) {
                    return message.likes.length;
                }
                message.likes.push(uid);
            }
            await this.col.updateOne(
                {
                    mid: {$eq: mid}
                },
                {
                    $set: {
                        likes: message.likes
                    }
                }
            );
        }
        return message.likes.length;
    }

    async hasLiked(mid, uid) {
        let message = await this.get(mid);
        if (message) {
            return message.likes.includes(uid);
        }
        return false;
    }

    async updateContent(mid, content) {
        await this.col.updateOne(
            { mid: { $eq: mid } },
            { $set: { 
                content: content, 
                editedAt: new Date()
            } }
        );
        return true;
    }    

}

exports.default = Messages;
