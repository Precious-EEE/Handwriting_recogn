# Handwriting Recognition

## Overview
This project focuses on recognizing handwritten digits using TensorFlow.js and the MNIST dataset. The implementation includes data preprocessing, model training, debugging, and memory optimization techniques to improve performance and accuracy.

## Features
- **Flattening Image Data**: Converts the MNIST handwritten digit images into a 1D array for processing.
- **Encoding Label Values**: Converts categorical labels into a numerical format and implements an accuracy gauge.
- **Debugging with Node.js**: Uses `node --inspect-brk` for step-by-step debugging of calculations.
- **Handling Zero Variance**: Identifies and manages cases where data has zero variance to prevent computation issues.
- **Memory Snapshots**: Uses the Chrome debugger to create memory snapshots for performance analysis.
- **Efficient Memory Management**: Releases references to the MNIST dataset to free up memory.
- **TensorFlow.js Optimization**: Implements `tf.tidy()` to manage TensorFlow.js memory usage effectively.
- **Footprint Reduction Measurement**: Evaluates memory consumption before and after optimizations.
- **Cost History & Model Accuracy Improvement**: Plots training cost history and optimizes the model for better accuracy.

