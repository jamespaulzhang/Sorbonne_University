document.getElementById("btn_ok").addEventListener("click", add);
function add (evt){
    let nombre1 = document.getElementById("nb1");
    let nombre2 = document.getElementById("nb2");
    let resultat = Number(nombre1.value)+Number(nombre2.value);
    console.log(resultat);
    if(evt.shiftKey)
        console.log("shift appuyé");
}