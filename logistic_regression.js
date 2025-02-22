const tf = require("@tensorflow/tfjs");

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.costHistory = [];

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
        batchSize: 10,
        decisionBoundary: 0.5,
      }, // fallback values if no options are provided
      options
    );
    // 1) Pick a value for 3 Ms and B
    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
    // [[0, 0, 0],
    //  [0, 0, 0],
    //  [0, 0, 0],
    //  [0, 0, 0]]
  }

  gradientDescent(features, labels) {
    // 2) Calculate cross entropy with respect to m and b
    // softmax equation: (e ^mx + b) / sum(e ^mx + b)
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels);
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    // 3) and 4) Multiply both slopes by learning rates and substract results from b and m
    return this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    // Handle batchs
    const { batchSize } = this.options;
    const batchQuantity = Math.floor(this.features.shape[0] / batchSize);
    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * batchSize;

        // Tidying the training loop: clean up unnecessary tensors from memory
        this.weights = tf.tidy(() => {
          const featureSlice = this.features.slice(
            [startIndex, 0],
            [batchSize, -1]
          );
          const labelSlice = this.labels.slice(
            [startIndex, 0],
            [batchSize, -1]
          );
          return this.gradientDescent(featureSlice, labelSlice);
        });
      }
      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1); // take the largest value on the horizontal axis
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);
    // Compare predictions to real labels and sum up the number of incorrect predictions
    const incorrect = predictions.notEqual(testLabels).sum().arraySync(); // 1s will be errors
    // Calculate error percentage
    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  processFeatures(features) {
    features = tf.tensor(features);
    // Standardize values
    // debugger;
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }
    // Add a column of 1s to enable matrix multiplication
    features = tf.ones([features.shape[0], 1]).concat(features, 1);
    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);
    // handle zero variances
    const filler = variance.cast("bool").logicalNot().cast("float32"); // 0s will be 1s
    this.mean = mean;
    this.variance = variance.add(filler);
    return features.sub(mean).div(this.variance.pow(0.5));
  }

  recordCost() {
    // Calculate vectorized cost
    // -(1/n) x (actual^T x log(guesses) + (- actual + 1)^T x log(- guesses + 1))
    const cost = tf.tidy(() => {
      const guesses = this.features.matMul(this.weights).sigmoid();
      const termOne = this.labels.transpose().matMul(guesses.add(1e-7).log());
      const termTwo = this.labels
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(guesses.mul(-1).add(1).add(1e-7).log()); // 1 x 10^-7 = 0.00000001 - add a tiny arbitrary constant to avoid log(0)
      return termOne
        .add(termTwo)
        .div(this.features.shape[0])
        .mul(-1)
        .arraySync()[0][0];
    });

    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }
    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2; // = this.options.learningRate / 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;