// logistic regression is used when you want to predict e.g. whether you have a cancerous tumor or not
// using some feature(s). the way we do this is by using a sigmoid function which looks a bit like
//
//                 _______________
//                /
//               /
//              /
// ____________/
//
// and it is h(x) = 1/(1 - e^-(mx + b)) (if only one feature) and in  general h(X) = 1/(1 - e^-(w_T X))
// and basically by looking at its base form h(z) = 1/(1 - e^-z) and plotting it we see that it has
// assymptotes at h = 0, 1. also when x = 0, h = 0.5, with no turning points, so we get the shape above.
// how do we get the w_T? well, we can define a cost function thus (remember h is the guess
// and y is the actual value):
//
// Cost(h, y) = - ln(h) if y = 1      (a)
//            = - ln(1 - h) if y = 0  (b)
//
// why? because in (a) if the predicted value is close to 1, then the error is close to 0. if the predicted_i
// value is close to 0, then the error is very large, thus 'penalizing' that guess. thus, we need to
// minimize the sum of all costs analogous to how we minimized the sum of all the distances from the
// actual value in linear regression. So, we have a mean error of
//
// J(w_T X) = 1/m * SUM_m Cost_m(h, y)
//
// and we can compact the cost function to
//
// Cost(h, y) = - y ln(h) - ( (1-y) ln(1 - h) )
//
// so if we minimize J, we differentiate to eventually get
//
// dJ/hX = 1/m * SUM_i X_i * (h(X_i) - y_i)
//
// note that this is the same as linear regression! we will thus multiply this by a learning rate and
// subract this from the weights' previous values. to compare accuracy, we use the predicted weights to
// run the sigmoid function, and compare that to the actual value.

require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const plot = require('node-remote-plot')
const LogisticRegression = require('./logistic-regression');

const TEST_DATA_RECORDS = 50;
const LEARNING_RATE = 0.5;
const ITERATIONS = 100;
const BATCH_SIZE = 10;
const DECISION_BOUNDARY = 0.5;

const { features, labels, testFeatures, testLabels } = loadCSV(
  '../data/cars.csv',
  {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    shuffle: true,
    splitTest: TEST_DATA_RECORDS,
    converters: {
      passedemissions: value => value === 'TRUE' ? 1 : 0
    } // maps true to 1 and false to 0
  }
);

const regression = new LogisticRegression(
  features,
  labels,
  {
    learningRate: LEARNING_RATE,
    iterations: ITERATIONS,
    batchSize: BATCH_SIZE,
    decisionBoundary: DECISION_BOUNDARY
  }
);

regression.train();
console.log(regression.test(testFeatures, testLabels));

plot({
  x: regression.costHistory.reverse()
});
