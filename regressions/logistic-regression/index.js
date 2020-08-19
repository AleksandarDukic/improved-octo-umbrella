require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LogisticRegression = require ('./logistic-regression');
const plot = require('node-remote-plot');
                                                        // pravi [[,,,],[,,,][,,,],...,[,,,]]
const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    dataColumns: [
        'horsepower',
        'displacement',
        'weight'
    ],
    labelColumns: [
        'passedemissions'
    ],
    shuffle: true,
    splitTest: 50,
    converters: {
        passedemissions: (value) => {
            return value === 'TRUE' ? 1 : 0;
        }
    }
});

/**
* @desc - Trains the model (Calculates the "a" and "b" coefficient of the d[f(x)]/d[a] and d[f(x)]/d[b]. f(x) = aX +b)
* @param array $features - Array (Array of Arrays) of all the features
* @param array $labels - Array of all the labels
* @param JSON $options - Options for the constructor
        $options:
        * @param number $learningRate
        * @param number $iterations - Number of epochs - runarounds through the whole Dataset
        * @param number $batchSize - Number of data samples in one Batch
        * @param number $decisionBoundary
*/
const regression = new LogisticRegression(features, labels, {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 10,
    decisionBoundary: 0.6
})

regression.train();

regression.test(testFeatures, testLabels);

regression.predict(testFeatures)
plot({
    x: regression.costHistory.reverse()
})