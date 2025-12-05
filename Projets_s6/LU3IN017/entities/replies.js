class Replies {

    constructor (db) {
        this.db = db;
        this.col = this.db.collection('replies');
    }

    async create (content, mid, uid) {
        let date = new Date();
        let reply = { content: content, date: date, mid: mid, uid: uid, likes: [] };

        let doc = await this.col.insertOne(reply);

        let strid = doc.insertedId.toString();
        await this.col.updateOne(
            {
                _id: {$eq: doc.insertedId}
            },
            {
                $set: {
                    rid: strid
                }
            }
        )
        return strid;
    }

    async get(rid) {
        return await this.col.findOne({
            rid: {$eq: rid}
        });
    }

    async getMany (params, sorting=+1) {
        return await this.col.find(params).sort({date: sorting}).toArray();
    }

    async delete (rid) {
        if (await this.get(rid) != null) {
            await this.col.deleteOne({
                rid: {$eq: rid}
            });
            return true;
        }
        return false;
    }

    async like (rid, uid, users) {
        let reply;
        if((reply = await this.get(rid)) != null) {
            let index;
            if((index = reply.likes.indexOf(uid)) > -1) {
                reply.likes.splice(index, 1);
            }
            else {
                if(await users.get(uid) == null) {
                    return reply.likes.length;
                }
                reply.likes.push(uid);
            }
            await this.col.updateOne(
                {
                    rid: {$eq: rid}
                },
                {
                    $set: {
                        likes: reply.likes
                    }
                }
            );
        }
        return reply.likes.length;
    }

    async hasLiked (rid, uid) {
        let reply = await this.get(rid);
        if (reply) {
            return reply.likes.includes(uid);
        }
        return false;
    }

}

exports.default = Replies;
