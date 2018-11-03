// we will use multinominal logistic regression to categorize numbers written on a 28 x 28 px
// grid. the way we will do this is by having each pixel be a feature with a grayscale value.
// we will flatten the grid into a single collection of numbers, each representing a pixel, each
// of which is a feature we will use to recognize handritten numbers from 1 -9. for this, we will
// use the mnist database, already loaded in the node modules.

// note that tensorflow keeps all references to every array inside of it, so every time you make
// a new tensor operation, it creates a new tensor and thus uses even more memory (tensors in tensor
// flow are immutable)- it will keep all references totensors and arrays created in every operation
// step so we need to do something to garbage collect what we will not need. in order to get around
// this, we use the tf method tf.tidy(() => { ... }) which basically will garbage collect everything
// except the return value. to inspect how many tensors tensorflow has stored one can simply run
// tf.ENV.registry.webgl.backend.texData.data and look at WeakMap -> entries
require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const _ = require('lodash');
const MultinominalLogisticRegression = require('./multinominal-logistic-regression');
const mnist = require('mnist-data');

const LEARNING_RATE = 1;
const ITERATIONS = 80;
const BATCH_SIZE = 500;
const DECISION_BOUNDARY = 0.5;
// loading many records can overload node, so we can custom its max space size like
// node --max-old-space-size=4096 image-recognizer.js
const TRAINING_SET_SIZE = 60000;
const TESTING_SET_SIZE = 10000;

// to call data from the mnist we go e.g., and we do it in a function because below mnistData
// takes up a lotta memory, so we want js' garbage collector to get rid of it once we distribute
// its data where it matters:
function loadData() {
  const mnistData = mnist.training(0, TRAINING_SET_SIZE); // starting at first arg, load second arg images
  // from which we get back something like
  // { images:
  //    { magic_number: 2051,
  //      total_num_items: 60000,
  //      rows: 28,
  //      cols: 28,
  //      values: [ [Array] ],
  //      start: 0,
  //      end: 1 },
  //   labels:
  //    { magic_number: 2049,
  //      total_num_items: 60000,
  //      values: [ 5 ],
  //      start: 0,
  //      end: 1 } }
  // you will notice by logging mnistData.images.values we are stil having an array of arrays,
  // with each inner array being a row in the 28 x 28 px image, so we want to flatten that
  const features = mnistData.images.values.map(image => _.flatMap(image));

  // now remember that our labels in the multinominal class are encoded as n rows, and column
  // number equal to the number of categories, which in this case is 10. when I call
  // mnistData.labels.values it just gives me an array of the actual number written, so we need
  // to encode that to a matrix.
  const encodedLabels = mnistData.labels.values.map(label => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
  });

  return { features, labels: encodedLabels };
}

const { features, labels } = loadData();

const regression = new MultinominalLogisticRegression(
  features,
  labels,
  {
    learningRate: LEARNING_RATE,
    iterations: ITERATIONS,
    batchSize: BATCH_SIZE
  }
);

regression.train();

const testMnistData = mnist.testing(0, TESTING_SET_SIZE);
const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));
const testEncodedLabels = testMnistData.labels.values.map(label => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

const accuracy = regression.test(testFeatures, testEncodedLabels);

console.log(`Accuracy is ${accuracy}`);

plot({
  x: regression.costHistory.reverse()
});
