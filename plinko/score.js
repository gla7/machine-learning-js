// k-nearest neighbor (KNN) is about asking ourselves what the dependent variable would
// likely be in a point in feature space, taking some readings of the dependent variable
// around that point, selecting the k nearest readings, and selecting the most common
// result. In this example we could ask: which bucket will a ball go to if dropped at e.g. 300px?
// We then drop a ball many times around the board and record which bucket each ball goes
// into. For each observation, we subtract drop point from 300px and take the absolute value
// so that we're looking at distance from the focus. We then sort results from least to
// greatest distance. We then take the top k records and see which is the most common bucket,
// and predict that for 300px, the ball will probably fall there. This is if we only keep
// dropPosition as a feature and no other featrues.

// How do we decide on the value of k? Well, one approach is to have a training set of data. We
// also have a test set. We run the analysis through the training set with various values of k,
// and for each one we test how good it predicts the test set (based in number of correctly
// predicted values). Basically the most accurate one wins.

const outputs = [];

// gather events in outputs variable
function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

// run an analysis to predict what k gives greatest accuracy
function runAnalysis() {
  const testSetSize = 100;
  const k = 10; // uncomment this line if you want to compare different features

  // const [testSet, trainingSet] = splitDataset(minMax(outputs, 3), testSetSize); // uncomment this line if you want to compare different k values

  // _.range(1, 25).forEach(k => { // uncomment this line if you want to compare different k
  _.range(0, 3).forEach(feature => { // uncomment this line if you want to compare different features
    // here for each k, for each of the testSet elements, we get a prediction of its end value based on
    // the training set, and compare
    const data = _.map(outputs, row => [row[feature], _.last(row)]); // uncomment this line if you want to compare different features
    const [testSet, trainingSet] = splitDataset(minMax(data, 1), testSetSize); // uncomment this line if you want to compare different features
    const accuracy = _.chain(testSet)
      .filter(testPoint => knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint)) // which predictions get it right
      .size() // get a count of them
      .divide(testSetSize) // divide by the total items to get an accuracy percentage
      .value()
    // console.log(`accuracy: ${accuracy} for k=${k}`); // uncomment this line if you want to compare different k values
    console.log(`accuracy: ${accuracy} for featrue ${feature}`); // uncomment this line if you want to compare different features
  })
}

// we will use data = trainingSet
function knn(data, point, k) {
  // initial gets all array elements but the last, and last gets the last
  return _.chain(data).map(row => {
      return [
        distance(_.initial(row), point),
        _.last(row)
      ]
    }) // only concerned with abs dist from desired feature space point and bucket
    .sortBy(row => row[0]) // sort data by abs distance
    .slice(0, k) // here we look at the closest k neighbors to prediction point
    .countBy(row => row[1]) // countBy returns an object with keys as the occurrence and value number of occurrences so that we select highest one
    .toPairs() // this just returns an array with arrays of two items, one for each of the key and value
    .sortBy(row => row[1]) // sort by occurrences
    .last() // choose the one with the highest number of occurrences
    .first() // get the corresponding bucket number
    .parseInt() // convert string to number
    .value(); // terminate this chain
}

// we use this function to check closeness of available observations to desired predictionPoint
function distance(pointA, pointB) { // pythagorean distance in n-dimensional space
  return _.chain(pointA)
  	.zip(pointB) // gathers same indeces in same array in the corresponding index
  	.map(([a, b]) => (a - b) ** 2) // set up pythagorean distance in each dimension
  	.sum() // get square of hypotenuse
  	.value() ** 0.5 // take square root
}

// here we split our observations into the testSet and trainingSet
function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);

  const testSet = _.slice(shuffled, 0, testCount); // takes from 0 to testCount
  const trainingSet = _.slice(shuffled, testCount); // takes from testCount to the end

  return [testSet, trainingSet];
}

// this function normalizes the data, so that the ranges of the features is always 0-1
function minMax (data, featureCount) { // feature count is to not normalize bucket label also!
  const clonedData = _.cloneDeep(data);

  // for each feature to be normalized (i is a column or feature)
  for (let i = 0; i < featureCount; i++) {
    const column = clonedData.map(row => row[i]); // we extract the corresponding column here

    const min = _.min(column); // determine min of that feature in dataset
    const max = _.max(column); // determine max of that feature in dataset

    // we iterate over each row (data point) and modify it so that it is normalized
    for (let j = 0; j < clonedData.length; j++) {
      clonedData[j][i] = (clonedData[j][i] - min) / (max - min); // we modify each value to normalize it
    }
  }

  return clonedData;
}

// NB: the ball bounciness, from observation, makes the result more unpredictable, so maybe
// it might not be a great feature considered. One way to make sure is to run the analysis
// using only one feature and then running the analysis using only another feature and then
// comparing the two.

// After carrying out the above analysis, it seems like the only useful feature (the one that
// has a greater chance of success than just chance) is drop position.
