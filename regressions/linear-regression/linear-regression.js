const tf= require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
    /* input : tensor, tensor, json */
    constructor(features, labels, options) {
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.mseHistory = [];
        
        this.options = Object.assign(
            { learningRate: 0.1, iterations: 10 },
            options                 //kopira options u {} objekat, tj. override-uje
        );

        this.weights = tf.zeros([this.features.shape[1], 1]);
        console.log("weights")
        this.weights.print();
    }

   /*  gradientDescent() {
        const currentGuessesForMPG = this.features.map(row => {
            return this.m * row[0] + this.b;                     // Array of data:  m*x(i) +b
        });

        const bSlope = 
            (
                _.sum(
                    currentGuessesForMPG.map((guess, i) => {
                        return guess - this.labels[i][0]
                    })
                ) *
            2) / 
            this.features.length;

        const mSlope = 
            (
                _.sum(
                    currentGuessesForMPG.map((guess, i) => {
                        return -1 * this.features[i][0] * (this.labels[i][0] - guess)
                    })
                ) * 
            2) / 
            this.features.length;
            console.log(mSlope)

        this.m = this.m - mSlope * this.options.learningRate;
        this.b = this.b - bSlope * this.options.learningRate;
    } */

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights)
        const differences = currentGuesses.sub( labels);

        const slopes = features
            .transpose()
            .matMul(differences)            // sabere slopove za b i m 
            .div(features.shape[0]);

        this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
    }


    train() {
        const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);

        for (let i = 0; i < this.options.iterations; i++) {
            for (let j =0; j < batchQuantity; j++) {
                const startIndex = j * this.options.batchSize;
                const { batchSize } = this.options;

                const featureSlice = this.features.slice(
                    [startIndex, 0],
                    [batchSize, -1]);

                const labelSlice = this.labels.slice(
                    [startIndex, 0],
                    [batchSize, -1]
                );

                this.gradientDescent(featureSlice, labelSlice);
            }
            this.recordMSE();
            this.updateLearningRate();
        }
    }

    predict(observations) {

        return this.processFeatures(observations).matMul(this.weights);
        
    }

    test(testFeatures, testLabels) {
        testFeatures = this.processFeatures(testFeatures);
        testLabels = tf.tensor(testLabels);

        const predictions = testFeatures.matMul(this.weights);

        predictions.print();

        const res = testLabels.sub(predictions)
            .pow(2)
            .sum()
            .get();
        
        const tot = testLabels.sub(testLabels.mean())
            .pow(2)
            .sum()
            .get();
        
        return 1 -res/tot;
    }


    /** 
    *   1) Turning an array of features into a Tensor
    *   2) Standardizes the Tensor
    *   3) Adding "1" in front
    *   @param features [array]
    */
    processFeatures(features) {
        console.log("lal1", features)
        features = tf.tensor(features);
        
        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5))
        } else {
            features = this.standardize(features);
        }
        features.print();
        features = tf.ones([features.shape[0], 1]).concat(features, 1);
        console.log("lal2");
        features.print();
        return features;
    }
    /** 
    *   1) Standardizes the Tensor
    *   @param features [tensor]
    */
    standardize(features) {
        const { mean, variance } = tf.moments(features, 0);

        this.mean = mean;
        this.variance = variance;

        return features.sub(mean).div(variance.pow(0.5))
    }

    recordMSE() {
        const mse = this.features
        .matMul(this.weights)
        .sub(this.labels)
        .pow(2)
        .sum()
        .div(this.features.shape[0])
        .get();

    this.mseHistory.unshift(mse)
    }

    updateLearningRate() {
        if (this.mseHistory.length < 2){
            return;
        }

        if (this.mseHistory[0] > this.mseHistory[1]) {
            this.options.learningRate /= 2;
        } else {
            this.options.learningRate *= 1.05
        }
    }
}

module.exports = LinearRegression;