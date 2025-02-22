// Reference: https://www.tensorflow.org/js/guide/nodejs
// require("@tensorflow/tfjs-node"); // CPU will be used for calculations
// require("@tensorflow/tfjs-node-gpu"); // GPU will be used for calculations
const tf = require("@tensorflow/tfjs");
const LogisticRegression = require("./multinominal-logistic-regression");
const _ = require("lodash");
const plot = require("nodeplotlib");
const mnist = require("mnist-data");

// Reduce memory usage
function loadData() {
  const mnistData = mnist.training(0, 60000); // up to 60 000
  const features = mnistData.images.values.map((image) => _.flatMap(image));
  const encodedLabels = mnistData.labels.values.map((label) => {
    const row = new Array(10).fill(0);
    row[label] = 1; // row[7] = 1 if label value is 7
    return row; // [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0  ],
  });
  return { features, labels: encodedLabels }; // Release reference to MNIST dataset
}

const { features, labels } = loadData();

const regression = new LogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 40,
  batchSize: 500, // batch quantity: 5000 / 100 = 50
});

regression.train();

const testMnistData = mnist.testing(0, 10000);
const testFeatures = testMnistData.images.values.map((image) =>
  _.flatMap(image)
);
const testEncodedLabels = testMnistData.labels.values.map((label) => {
  const row = new Array(10).fill(0);
  row[label] = 1; // row[7] = 1 if label value is 7
  return row; // [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0  ],
});

const accuracy = regression.test(testFeatures, testEncodedLabels);
console.log("Accuracy is", accuracy); // 0.9253

// Plot data
const data = [
  {
    y: regression.costHistory.reverse(),
  },
];
const layout = {
  title: {
    text: "Plotting Cost History",
    font: {
      family: "Courier New, monospace",
      size: 24,
    },
    xref: "paper",
  },
  xaxis: {
    title: {
      text: "Iteration #",
      font: {
        family: "Courier New, monospace",
        size: 18,
        color: "#7f7f7f",
      },
    },
  },
  yaxis: {
    title: {
      text: "Cost",
      font: {
        family: "Courier New, monospace",
        size: 18,
        color: "#7f7f7f",
      },
    },
  },
};

plot.plot(data, layout);