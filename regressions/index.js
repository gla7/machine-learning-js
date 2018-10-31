// in this section we will implement gradient descent algorithm in order to make
// a linear regression to predict mpg of cars using the horsepower (initially). the
// way gradient descent works is:
//
// 1. first we look at the data points in n-dimensional space
// 2. if it seems like a line could describe more or less the positions of
//    these points in n-d space then we need to determine what that line looks
//    like (y = mx + b where in our example x would be horsepower and y would
//    be mpg)
// 3. the average squared error from the prediction line and the actual values
//    is logically E = 1/n SUM_i(mx_i + b - value_i)^2
// 4. we want our prediction line to minimize this error, i.e. dE/db and dE/dm
//    close to zero
// 5. dE/db = 2/n SUM_i(mx_i + b - value_i) and
//    dE/dm = 2/n SUM_i(x_i(mx_i + b - value_i)) noting that in matrix form
  // if w = (m) is my weights matrix, and x =(x_11 x_01)
  //        (b)                              (..     ..)
  //                                         (x_1n x_0n) is the variables matrix, so we
  // could write dE/dw = 1/n * (x^T * ((x * w) - labels)), which has n rows and
  // 1 column (x_0 is always 1 and x_1 is what we formerly knew as x)
// 6. we follow this algorithm: we set an initial arbitrary value for both m and
//    b, then see what the resulting dE/db and dE/dm are
// 7. repeat step 6 but with m = m - alpha * dE/dm and b = b - alpha * dE/db,
//    where alpha is called the learning rate, which should be low to start with
// 8. repeat step 7 until dE/dm and dE/db are as close to zero as possible and
//    not changing very much on each step (you must decide the criteria for
//    stopping)
// 9. record the last values of m and b and these will give you the formula for
//    your prediction line, which you can use to make predictions with a given x
//    (which can be higher than 1-d)

// in order to gauge the accuracy of our model we will use something known as coefficient
// of determination. it is defined as R^2 = 1 - SS_res / SS_tot, where SS_res and SS_tot
// are sum of squares of residuals and total sum of squares respectively:
// SS_res = SUM_i (actual_i - predicted_i)^2
// SS_tot = SUM_i (actual_i - average_label)^2
// so we can see that for a good fit, SS_res will be much smaller than SS_tot, which would
// mean a value of R^2 close to zero. if, however, SS_tot is smaller than SS_res, then
// R^2 can be negative, whoch would indicate a bad fit of our prediction line. basically
// a negative coefficient of determination means we're better off predicting using the mean

// also recall that one useful method of feature scaling is normalization, whereby we
// ask the question: how many standard deviations away from the mean is this value? i.e.
// (feature - mean) / stdv and importantly, we use the same mean and stdv in test and
// training sets!

// another thing that we will do is to have a dynamic learning rate, i.e. a learning rate that
// adjusts itself according to how well it's doing. though there are several popular methods to
// do this, we're going to implement a simplistic version of our own. the way it will work will be:
//
// 1. on every iteration of gradient descent record the mean squared error (MSE)
// 2. next iteration after that, record MSE again
// 3. compare both MSEs
// 4. if MSE got bigger, then reduce learning rate by half (this is arbitrary)
// 5. if MSE got smaller, increase learning rate by 5%

require('@tensorflow/tfjs-node');

const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

const TEST_DATA_RECORDS = 50;
const LEARNING_RATE = 0.1;
const ITERATIONS = 100;

let { features, labels, testFeatures, testLabels } = loadCSV(
  './cars.csv',
  {
    shuffle: true,
    splitTest: TEST_DATA_RECORDS,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
  }
);

const regression = new LinearRegression(
  features,
  labels,
  {
    learningRate: LEARNING_RATE,
    iterations: ITERATIONS
  }
);

regression.train();
const r2 = regression.test(testFeatures, testLabels);

// let's plot the progression of our dynamic learning rate
plot({
  x: regression.mseHistory.reverse(), // so that it plots the right way round since we unshifted to this array
  xLabel: 'iteration number',
  yLabel: 'mean squared error'
});

console.log(`R^2 is ${r2}`)
