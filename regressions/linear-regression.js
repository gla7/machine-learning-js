const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

const DEFAULT_LEARNING_RATE = 0.1;
const DEFAULT_NO_OF_ITERATIONS = 1000;

class LinearRegression {
  constructor(features, labels, options) {
    // slow but explicit approach
    // this.features = features;
    // this.labels = labels;
    // fast version using tesnorflow
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.options = Object.assign({
      learningRate: DEFAULT_LEARNING_RATE,
      iterations: DEFAULT_NO_OF_ITERATIONS
    }, options); // copies the options key-value pairs and merges them with {}
    // slow but explicit approach
    // this.m = 0;
    // this.b = 0;
    // fast version using tesnorflow
    this.weights = tf.zeros([this.features.shape[1], 1]);
    this.mseHistory = []; // to see how to adjust learning rate, perhaps could just be 2 values, last and current
  }

  gradientDescent() {
    // slow but explicit approach
    // const currentGuessesForMPG = this.features.map(row => {
    //   return this.m * row[0] + this.b; // mx_i + b
    // });
    //
    // const bSlope = _.sum(currentGuessesForMPG.map((guess, index) => {
    //   return guess - this.labels[index][0] // mx_i + b - mpg_i
    // })) * (2 / this.features.length);
    //
    // const mSlope = _.sum(currentGuessesForMPG.map((guess, index) => {
    //   return this.features[index][0] * (guess - this.labels[index][0]) // x_i(mx_i + b - mpg_i)
    // })) * (2 / this.features.length);
    //
    // this.b = this.b - this.options.learningRate * bSlope;
    // this.m = this.m - this.options.learningRate * mSlope;
    // fast version using tesnorflow
    const currentGuesses = this.features.matMul(this.weights); // m_jx_i + bx_0
    const differences = currentGuesses.sub(this.labels); // m_jx_i + bx_0 - actual_i

    // 2/n SUM_i(x_i_t(m_jx_i + bx_0 - actual_i)) BUT we don't include the 2 since this derivative
    // is used for learning rate purposes only:
    const gradients = this.features
                       .transpose()
                       .matMul(differences)
                       .div(this.features.shape[0]);

    this.weights = this.weights.sub(gradients.mul(this.options.learningRate));
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
      this.recordMSE();
      this.updateLearningRate()
    }
  }

  test(testFeatures, testLabels) { // tests my prediction line against test data
    // processing to tensors, see index.js for explanation of coefficient of determination
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights); // this computes line equation for each data point

    const SS_res = testLabels.sub(predictions).pow(2).sum().get();
    const SS_tot = testLabels.sub(testLabels.mean()).pow(2).sum().get();

    return 1 - (SS_res / SS_tot); // the coefficient of determination
  }

  processFeatures(features) {
    features = tf.tensor(features);
    // important to standardize before adding column of ones otherwise that gets standardized too!
    features = this.standardize(features);
    features = tf.ones([features.shape[0], 1]) // single column, n rows of ones or x_0
                 .concat(features, 1);

    return features;
  }

  standardize(features) {
    // we do not want to re-calculate mean and variance if we have them
    if (this.mean && this.variance) return features.sub(this.mean).div(this.variance.pow(0.5));
    const { mean, variance } = tf.moments(features, 0); // gets moments for each row of data

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5)); // how many stdv is each value away from the mean
  }

  recordMSE() { // this method is to see how we should adjust the learning rate on each gradient descent step
    const mse = this.features.matMul(this.weights).sub(this.labels).pow(2).sum().div(this.features.shape[0]).get();
    this.mseHistory.unshift(mse); // put the last value at the front
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) return;
    const lastValue = this.mseHistory[0];
    const previousValue = this.mseHistory[1];
    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2; // divide by 2 if our estimate is getting worse
    } else {
      this.options.learningRate *= 1.05; // increase by 5% if our estimate is getting better to accelerate learning
    }
  }
}

module.exports = LinearRegression;
