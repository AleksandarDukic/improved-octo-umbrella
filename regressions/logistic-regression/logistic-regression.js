const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

/**
 * @desc constructor for LogisticRegression. Loads features (Array or Array of arrays), labels (array), options (JSON). Makes standard tensors out of
 * features and labels. Makes options object - overriding the defaults. Makes new tensor - weights[1,2] - initialized with zeros
 * @param array $features - Array (Array of Arrays) of all the features
 * @param array $labels - Array of all the labels
 * @param JSON $options - Options for the constructor
 * @return Class - tensorFlow like class
 */

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = [];
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 10, decisionBoundary: 0.5 },
      options //kopira options u {} objekat, tj. override-uje
    );

    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  /**
   * Monitoring the first Derivate of MSE() : 1/N ∑ (b + m1x° + m2x¨ + m3xº - Y()) ^ 2
   * @desc Performs a gradientDescent operation on one batch of $features/$labels
   * @param tensor $features - takes in batchSize of features
   * @param tensor $labels   - takes in batchSize of labels
   */
  gradientDescent(features, labels) {
    // [10,4] x [4,1] = [10,1] = a + bx1 + cx2 + dx4 ... sum
    const currentGuesses = features.matMul(this.weights).sigmoid(); // Formula for calculating sigmoid(x): y = 1 / (1 + exp(-x)). Returns (0,1)
    const differences = currentGuesses.sub(labels); // sum(guess) - label [10,1]
    const slopes = features // [10,4]
      .transpose() // [4,10]
      .matMul(differences) // [4,10] X [10,1] - [4,1] summs slopes for b and m1, m2, m3
      .div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  /**
   * @desc - Trains the model (Calculates the "a" and "b" coefficient of the d[f(x)]/d[a] and d[f(x)]/d[b]. f(x) = aX +b)
   *         For a slice of features and labels perfroms gradientDescent(featureSlice, labelSlice)
   *         saves Data
   *         updates Learning rate
   */
  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    ); // [number of data samples] / [size of a batch]

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;

        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        ); // [10,4] - matrix - (-1 means to take in consideration all the columns till the far right end)
        const labelSlice = this.labels.slice(
          [startIndex, 0],
          [batchSize, -1] // [10,1]
        );

        this.gradientDescent(featureSlice, labelSlice);
      }
      this.recordCost();
      this.updateLearningRate();
    }
  }

  /**
   * -----------
   */
  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .sigmoid()
      .greater(this.options.decisionBoundary)
      .cast("float32");
  }

  /**
   * ----------
   */
  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);

    testLabels = tf.tensor(testLabels);

    const incorrect = predictions
      .sub(testLabels)
      .abs()
      .sum()
      .get();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  /**
   * @desc - Makes a tensor out of an array(array of arrays). Standardizes the tensor. Adds a column of ones in front of the "features" matrix
   * @param array $features
   * @return tensor - standardized tensor made out of an array
   */
  processFeatures(features) {
    features = tf.tensor(features);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  /**
   * @desc - Calculates "mean" and "variance" parameters based on the input tensor
   * @param tensor $features
   * @return tensor $features - standardized tensor
   */
  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  /**
   * !!! This function usess CROSS-ENTROPY to update Learning Rate so we DO NOT fall into a local minimum, but find the real MINIMUM   !!!
   * This is a CrossEntropy Formula
   */
  recordCost() {
    const guesses = this.features.matMul(this.weights).sigmoid();
    const termOne = this.labels.transpose().matMul(guesses.log());

    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(
        guesses
          .mul(-1)
          .add(1)
          .log()
      );

    const cost = termOne
      .add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .get(0, 0);   // pulls out the single value , cancels the dimmensions [[x]] -> x

    this.costHistory.unshift(cost);
  }

  /**
   * Changing the LearningRate +/- in regard to the CostHistory
   */
  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;

/* 
Batch size is a term used in machine learning and refers to the number of training examples utilised in one iteration. The batch size can be one of three options:

batch mode: where the batch size is equal to the total dataset thus making the iteration and epoch values equivalent
mini-batch mode: where the batch size is greater than one but less than the total dataset size. Usually, a number that can be divided into the total dataset size.
stochastic mode: where the batch size is equal to one. Therefore the gradient and the neural network parameters are updated after each sample.
 */

/* 
 
 An Epoch represents one iteration over the entire dataset.

 We cannot pass the entire dataset into the neural network at once. So, we divide the dataset into  number of batches.

 If we have 10,000 images as data and a batch size of 200, then an epoch should contain 10,000 / 200 = 50 iterations*. (iteration is a one procedure over the whole dataset)
 
 */
