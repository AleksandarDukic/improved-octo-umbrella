require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require ('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data') // by default it has 2 parts of data: training( 60000 images ), test( 20000 images )

const mnistData = mnist.training(0, 1000);

const features = mnistData.images.values.map(image => _.flatMap(image))
const encodedLabels = mnistData.labels.values.map(label => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
})

const regression = new LogisticRegression(features, encodedLabels, {
    learningRate: 1,
    iterations: 5,
    batchSize: 100
})

regression.train();

const testMnistData = mnist.testing(0,100);
const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map(label => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
})

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log("accuracy is ",accuracy)

// node --inspect-brk index.js

// otvorimo Google Chrome na stranu 

//  chrome://inspect  --> kliknemo na "inspect"

// imali smo gresku kod :
// features = features.sub(this.mean).div(this.variance.pow(0.5));
// dobija se [NaN,....NaN] jer delimo sa nulom
// browser pretvara deljenje sa nulom u 1
// ali kod noda se pretvara u Infinity pa u NaN

// jedan nacin je da se izbace sve 0 iz features objekta - "Remove a COLUMN from features values with all 0s"
// drugi je da promenimo standardizaciju - sto cemo i uraditi
// gde god u "variance" tenzoru vidimo 0 pretvaramo ga rucno u 1:
// variance = variance.cast('bool').logicalNot() + variance -- popunjavamo nule sa jedinicama