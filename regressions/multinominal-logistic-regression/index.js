// after binary classification, we will implement multinominal classification to classify into various categories
// and lets take the following example: given a persons age, do they prefer to read, watch movies, or dance? to
// approach this, we have 2 options:
//
// 1. rather than saying (as before) that if the sigmoid is below 0.5 then watching movies else
//    reading, now we will have three sigmoids: one for watching movies, another for reading and another for dancing,
//    each of these 0-1. so its like having 3 labels. so 3 logistic regressions. then we take each data point (in this
//    case a person) and we will have 3 labels for each and 3 predictions. we could just take the greatest one and
//    predict that thats what the person wants to do.
//
// 2. if you look at approach 1, we can notice that the features will be exactly the same for the 3 logistic
//    regressions. however, the weights and the labels do turn out to be different. what if we were to combine
//    our weights tensors for each category into one, and similar with the features tensors? would we be able
//    to do this? well we have (for one feature on each) weights and encoded labels respectively (remember,
//    there are two labels for each classification because of the x_0 which is always 1) for watching movies,
//    reading books, and dancing respectively:
//
//    (b_1)     (1)     (b_2)     (0)     (b_3)     (0)
//    (m_1)     (1)     (m_2)     (0)     (m_3)     (0)
//              (0)               (1)               (0)
//              (0)               (1)               (0)
//              (0)               (0)               (1)
//              (0)               (0)               (1)
//
//    so we can see that a combined weights tensor and a combined labels tensor would maybe be
//
//    (b_1 b_2 b_3)     (1 0 0)
//    (m_1 m_2 m_3)     (1 0 0)
//                      (0 1 0)
//                      (0 1 0)
//                      (0 0 1)
//                      (0 0 1)
//
//    does the math still work? we are aiming for this equation to still work:
//    dJ/hX = 1/m * SUM_i X_i_T * (h(X_i w) - y_i) so for X_i (features) * w (weights) matrix multiplication
//    for this example, remembering that my features tensor is n by 2, we get an n by 3 tensor, or in this
//    example:
//
//    (1 x_1)                                              (m_1x_1+b_1 m_2x_1+b_2 m_3x_1+_b_3)
//    (1 x_1)                                              (m_1x_1+b_1 m_2x_1+b_2 m_3x_1+_b_3)
//    (1 x_1)   which when multiplied by the weights gives (m_1x_1+b_1 m_2x_1+b_2 m_3x_1+_b_3)
//    (1 x_1)                                              (m_1x_1+b_1 m_2x_1+b_2 m_3x_1+_b_3)
//    (1 x_1)                                              (m_1x_1+b_1 m_2x_1+b_2 m_3x_1+_b_3)
//    (1 x_1)                                              (m_1x_1+b_1 m_2x_1+b_2 m_3x_1+_b_3)
//
//    so that is the same size as my labels tensor and it is in fact what I would like to take a sigmoid of and
//    compare to the labels! the rest of the dJ/hX does not change. thus, doing this approach basically amounts
//    to rolling the 3 logistic regressions into a single equation. to then make a prediction, we will just
//    take our prediction features, multiply by the combined weights, and take the sigmoid, then select the
//    row with the greatest value for each row (data point).

// concretely here we will solve: given the horsepower, weight, and displacement of a vehicle, will it have
// high, medium, or low fuel efficiency? to categorize the fuel efficiency, we use mpg: 0-15 low, 15-30 med,
// 30+ high

// note that the approach 2. above will have an issue: we will be calculating probabilities (from the sigmoid)
// that are not conditional to the other two events NOT happening, and so its possible to obtain a prediction
// that both says that the person wants to watch movies AND dance for example. this makes sense for watching
// movies and dancing, but not for the categorized fuel efficiency above because they are contradictory. to
// remedy this we need to use conditional probabilities, i.e. what is the probability of having low fuel
// efficiency AND not having medium OR high? so far weve been using the sigmoid function because it has
// the convenient behavior to go to 0 or 1, intersecting the y-axis at z = 0. it turns out that the sigmoid
// function is just a special case of a function that has the same convenient properties: the softmax function.
// a good derivation of the softmax can be found in http://willwolf.io/2017/04/19/deriving-the-softmax-from-first-principles/
// which i also attached in these files in case his page goes down (gotta keep it) and ive also attached
// a reviewed justification of the conditional probability and the softmax function properties.
// basically the softmax function describes the conditional probability of an event occurring whilst other events
// occur also (in our example mpg being low and not medium or high, etc).

require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const plot = require('node-remote-plot');
const _ = require('lodash');
const MultinominalLogisticRegression = require('./multinominal-logistic-regression');

const TEST_DATA_RECORDS = 50;
const LEARNING_RATE = 0.5;
const ITERATIONS = 100;
const BATCH_SIZE = 10;
const DECISION_BOUNDARY = 0.5;

const { features, labels, testFeatures, testLabels } = loadCSV(
  '../data/cars.csv',
  {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['mpg'],
    shuffle: true,
    splitTest: TEST_DATA_RECORDS,
    converters: {
      mpg: value => {
        const mpg = parseFloat(value);

        // here we make the categorization of the mpg label
        // note, though that because parseCSV always returns an array of arrays
        // the dataset of labels will come back like [[[0, 1, 0]], [[1, 0, 0]], ...]
        // so are one level too deeply nested, so below we will use lodash to
        // remedy this- flatMap removes one level of nesting inside of the array called
        if (mpg < 15) {
          return [1, 0, 0];
        } else if (mpg < 30) {
          return [0, 1, 0];
        } else {
          return [0, 0, 1];
        }
      }
    } // maps true to 1 and false to 0
  }
);

const regression = new MultinominalLogisticRegression(
  features,
  _.flatMap(labels),
  {
    learningRate: LEARNING_RATE,
    iterations: ITERATIONS,
    batchSize: BATCH_SIZE,
    decisionBoundary: DECISION_BOUNDARY
  }
);

regression.train();
console.log(regression.test(testFeatures, _.flatMap(testLabels)));
