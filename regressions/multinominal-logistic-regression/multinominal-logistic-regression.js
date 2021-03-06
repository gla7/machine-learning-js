const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

const DEFAULT_LEARNING_RATE = 0.1;
const DEFAULT_NO_OF_ITERATIONS = 1000;
const DEFAULT_DECISION_BOUNDARY = 0.5;

class MultinominalLogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.options = Object.assign({
      learningRate: DEFAULT_LEARNING_RATE,
      iterations: DEFAULT_NO_OF_ITERATIONS,
      decisionBoundary: DEFAULT_DECISION_BOUNDARY
    }, options);
    // here we use the combined labels tensor as the number of weights columns
    // as per the explanation in ./index.js
    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
    this.costHistory = [];
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels);

    const gradients = features
                       .transpose()
                       .matMul(differences)
                       .div(features.shape[0]);

    return this.weights.sub(gradients.mul(this.options.learningRate));
  }

  train() {
    const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);
    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = this.options.batchSize * j;
        const { batchSize } = this.options;

        // we do not assign the weights here anymore because we have wrapped this in a tf.tidy in train()
        // so as to not keep all the tensors generated in every loop, so here we return the updated weights
        // which are returned in turn in the tf.tidy in train, and this is assigned to the updated weights
        this.weights = tf.tidy(() => {
          const featuresSlice = this.features.slice([startIndex, 0], [batchSize, -1]);
          const labelsSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);

          return this.gradientDescent(featuresSlice, labelsSlice);
        });
      }
      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations, decisionBoundary) {
    return this.processFeatures(observations)
               .matMul(this.weights)
               .softmax()
               .argMax(1); // get column index of the largest value on horizontal axis so
                           // that our prediction is simply an index
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);

    // note that here we use notEqual which compares two tensors and in the i, j th position
    // it puts a 0 if i, j th are equal in both and 1 if not equal, then we're gonna sum
    // as we will always get only 1 incorrect, so that will be our number of incorrect guesses
    const incorrect = predictions.notEqual(testLabels).sum().get();

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

    // if variances are zero, features.sub(mean).div(variance.pow(0.5)) becomes an issue,
    // so we deal with that by changing zeroes to ones using logicalNot method
    const filler = variance.cast('bool').logicalNot().cast('float32');

    this.mean = mean;
    this.variance = variance.add(filler); // it will add the zeroes to ones from filler

    return features.sub(mean).div(this.variance.pow(0.5));
  }

  recordCost() {
    const cost = tf.tidy(() => {
      // cost function can keep the sigmoid since it still describes the penalty for
      // not getting a guess right
      const guesses = this.features.matMul(this.weights).sigmoid();
      // see note below for the add 1e-7
      const termOne = this.labels.transpose().matMul(guesses.add(1e-7).log());
      // note that we add 1e-7 in case any of our guesses rounds up to 1, since we then
      // multiply by -1 and then subtract 1 and then take the log... this would result in -infty.
      // since the guesses can never quite be zero (assyptotical value), after we add a small
      // number that ensures the log will never be log of zero
      const termTwo = this.labels.mul(-1).add(1).transpose().matMul(guesses.mul(-1).add(1).add(1e-7).log());

      return termOne.add(termTwo).div(this.features.shape[0]).mul(-1).get(0, 0);
    });

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

module.exports = MultinominalLogisticRegression;
