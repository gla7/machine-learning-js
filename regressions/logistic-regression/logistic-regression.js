const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

const DEFAULT_LEARNING_RATE = 0.1;
const DEFAULT_NO_OF_ITERATIONS = 1000;
const DEFAULT_DECISION_BOUNDARY = 0.5;

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.options = Object.assign({
      learningRate: DEFAULT_LEARNING_RATE,
      iterations: DEFAULT_NO_OF_ITERATIONS,
      decisionBoundary: DEFAULT_DECISION_BOUNDARY
    }, options);
    this.weights = tf.zeros([this.features.shape[1], 1]);
    this.costHistory = [];
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).sigmoid();
    const differences = currentGuesses.sub(labels);

    const gradients = features
                       .transpose()
                       .matMul(differences)
                       .div(features.shape[0]);

    this.weights = this.weights.sub(gradients.mul(this.options.learningRate));
  }

  train() {
    const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);
    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = this.options.batchSize * j;
        const { batchSize } = this.options;

        const featuresSlice = this.features.slice([startIndex, 0], [batchSize, -1]);
        const labelsSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

        this.gradientDescent(featuresSlice, labelsSlice);
      }
      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations, decisionBoundary) {
    // we apply the test boundary with round
    return this.processFeatures(observations)
               .matMul(this.weights)
               .sigmoid()
               .greater(this.options.decisionBoundary)
               .cast('float32'); // to convert from bool to num
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels);

    // here is our metric of quality: taking the absolute diff between predictions and labels
    // and then summing them, this will tell me how many were incorrect
    const incorrect = predictions.sub(testLabels).abs().sum().get();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  processFeatures(features) {
    features = tf.tensor(features);
    features = this.standardize(features);
    features = tf.ones([features.shape[0], 1])
                 .concat(features, 1);

    return features;
  }

  standardize(features) {
    if (this.mean && this.variance) return features.sub(this.mean).div(this.variance.pow(0.5));
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }

  recordCost() {
    // here we need - 1/m SUM_i ( y_i ln(h) + (1 - y_i) ln(1 - h) ) which is my cost function
    const guesses = this.features.matMul(this.weights).sigmoid();
    const termOne = this.labels.transpose().matMul(guesses.log());
    const termTwo = this.labels.mul(-1).add(1).transpose().matMul(guesses.mul(-1).add(1).log());
    const cost = termOne.add(termTwo).div(this.features.shape[0]).mul(-1).get(0, 0);
    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) return;
    const lastValue = this.costHistory[0];
    const previousValue = this.costHistory[1];
    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
