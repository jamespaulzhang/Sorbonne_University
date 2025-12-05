const path = require('path');
const api = require('./api.js');
const { MongoClient } = require('mongodb');

// Détermine le répertoire de base
const basedir = path.normalize(path.dirname(__dirname));
console.debug(`Base directory: ${basedir}`);

const express = require('express');

const app = express();

const session = require('express-session');

app.use(session({
    secret: 'technoweb rocks',
    resave: true,
    saveUninitialized: false
}));

const dbName = 'forum';
const client = new MongoClient('mongodb://localhost:27017');

async function connectMongoDb() {
    try {
        // Connexion à la base de données
        await client.connect();
        console.log("Connexion avec la base de données");
        console.log("");
    }
    catch(e) {
        console.error(e);
    }
}

connectMongoDb().catch(console.error);

// Lancement de l'api
app.use('/api', api.default(client.db(dbName)));

app.on('close', async () => {
    await client.close();
});

exports.default = app;
