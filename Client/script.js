const url = 'http://90.156.204.192:5000/predict';

var button = document.getElementById("button");
var result = document.getElementById("result");

button.addEventListener('click', async () => {
    var text = document.getElementById("text").value;
    if (text == ''){
        result.innerHTML = "";
        alert("Enter text!");
    } else {
        var response = await fetch(`${url}?text="${text}"`, {method: 'get'})
            .then(resp => resp.text())
            .then(data => { console.log(data); result.innerHTML = data;})
            .catch(err => { console.log(err) });

        
    }
});