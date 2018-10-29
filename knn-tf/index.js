// we require the first which instructs tensorflow to perform calculations on your cpu
// you can also have it do calculations on your gpu by requiring '@tensorflow/tfjs-node-gpu'
// as well as a couple other libraries
require('@tensorflow/tfjs-node');

const tf = require('@tensorflow/tfjs'); // standard tf library
const loadCSV = require('./load-csv'); // library to load csvs on js (adjacent file)

const TEST_DATA_RECORDS = 10;
const k = 10;

// for an explanation of tensorflow and this modified algorithm, see my intro
// to tensorflow js repo
function knn(features, labels, predictionPoint, k) {
  // to see how we applied standarization in more detail see my intro to tf js repo
  const { mean, variance } = tf.moments(features, 0);

  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

  return features
    .sub(mean)
    .div(variance.pow(0.5))
    .sub(scaledPrediction)
    .pow(2)
  	.sum(1)
  	.pow(0.5)
  	.expandDims(1)
    .concat(labels, 1)
    .unstack()
  	.sort((a, b) => a.get(0) > b.get(0) ? 1 : -1)
  	.slice(0, k)
  	.reduce((acc, pair) => {
  		return acc + pair.get(1);
  	}, 0)/k;
}

// note that especially with sqft_lot there are massive properties that are really
// outliers. if we were to use normalization (values between 0 and 1) the massive
// properties could make most values gather close to zero whilst few values would
// stay at or near 1. This data could be better modelled via standarization, which
// is made to reflect the nature of a normal distribution, where the majority of the
// data is within a standard deviation. To standardize we go (value - avg) / stdv,
// and tf has tools that make this easy.
let { features, labels, testFeatures, testLabels } = loadCSV(
  'kc_house_data.csv',
  {
    shuffle: true,
    splitTest: TEST_DATA_RECORDS,
    dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
    labelColumns: ['price']
  }
);

// currently these values are given as js arrays
features = tf.tensor(features);
labels = tf.tensor(labels);

// testFeatures are the desired predictions' values (like lat + long), and
// testLabels are the actual value we're trying to predict
testFeatures.forEach((testPoint, index) => {
  const result = knn(features, labels, tf.tensor(testPoint), k);
  const err = (testLabels[index][0] - result) / testLabels[index][0];
  console.log(`Our accuracy for test point ${index + 1} is ${err * 100}%`)
});

// note that we can debug our code by running e.g. node --inspect-brk index.js and then
// going to chrome and going to chrome://inspect and clicking inspect on index.js process

// further improvements could be done with k value analysis and feature analysis
