const app = require('./app.js');

const port = 4000;

app.default.listen(port, () => {
    console.log("");
    console.log("=======================================")
    console.log(`Serveur actif sur le port ${port}`);
    console.log("=======================================")
    console.log("");
});
