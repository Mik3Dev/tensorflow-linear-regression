import 'bootstrap/dist/css/bootstrap.css';
import * as tf from '@tensorflow/tfjs';

/*
    Definicion del modelo para la regresion lineal:
    
    El método tf.sequential() crea un nuevo modelo. El modelo sequential (secuencial) 
    es cualquier modelo donde todos los outputs (salidas) de una capa son los inputs 
    (entradas) de la siguiente capa.

    Ya creado el modelo, se agrega la primera capa con el comando model.add(),
    la nueva capa llama al método tf.layers.dense(). Se crea una capa densa, la cual 
    cada nodo de esta capa estara conectada con todos los nodos de la siguiente capa.
*/
const model = tf.sequential();
model.add(tf.layers.dense({
    units: 1,
    inputShape: [1]
}));

/*
    Seguido se llama al método model.compile(), el cual se encarga de preparar al modelo
    para el entrenamiento y evaluación. El parametro loss representa la función de error
    para este caso el meanSquaredError. El parametro optimizer que representa la función
    de optimizacion para este caso se usa sgd (Stochastic Gradident Descent - Gradiente 
    Descendiente Estocástico). 
*/
model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
});

/* 
    Creamos un par de tensores para los datos de entrenamiento. El tensor xs con valores de 0 a 10, 
    como valores de entrada (input). El tensor ys con los valores deseados y que cumplen 
    con la siguiente función.

    ys = 3 * xs + 5
*/
const xs = tf.linspace(0, 10, 11);
const ys = xs.mul(3).add(5);

const xsData = xs.dataSync();
const ysData = ys.dataSync();

let body = ''
for(let i=0; i<xsData.length; i++){
    body += `<tr>
        <td>${i+1}</td>
        <td>${xsData[i]}</td>
        <td>${ysData[i]}</td>
    </tr>`
}
document.getElementById('tbody').innerHTML = body


/*
    Entrenamiento del modelo, una vez creado el modelo y los valores de entrenamiento con el 
    metodo model.fit()
*/
model.fit(xs, ys, {
    epochs: 1000,
}).then(() => {
    const btn = document.getElementById("predictBtn");
    btn.disabled = false;
    btn.innerText = "Calcular";
    
    btn.addEventListener('click', (e) => {
        const value = parseFloat(document.getElementById('inputValue').value);
        const predictedTensor = model.predict(tf.tensor([value]));
        const predicted = predictedTensor.dataSync();
        document.getElementById('output').innerText = predicted[0];
    });

    document.getElementById('predictedFormula').classList.remove('d-none')
    document.getElementById('w1').innerText = model.getWeights()[0].dataSync()
    document.getElementById('b1').innerText = model.getWeights()[1].dataSync()
});


