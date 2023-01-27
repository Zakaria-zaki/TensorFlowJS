const tf = require('@tensorflow/tfjs');
const tfnode = require('@tensorflow/tfjs-node');
const fs = require('fs');
const labels = require('./labels.js');

const modelUrl = 'https://tfhub.dev/tensorflow/tfjs-model/ssdlite_mobilenet_v2/1/default/1';

let model;

// load our model from tensorFlow hub
const loadModel = async () => {
    console.log('Loading model ...');

    model = await tf.loadGraphModel(modelUrl, { fromTFHub: true });
   
    console.log('model load with success');


    return model;
}

// perform non maximum suppression of bounding boxes
const calculateNMS = function (outputBoxes, maxScores) {
    console.log('calculating box indexes');
  
    const boxes = tf.tensor2d(outputBoxes.dataSync(), [outputBoxes.shape[1], outputBoxes.shape[3]]);
    const indexTensor = tf.image.nonMaxSuppression(boxes, maxScores, maxNumBoxes, 0.5, 0.5);
  
    return indexTensor.dataSync();
  }

// convert image to Tensor using decode image API
const processInput = function (imagePath) {
    console.log(`preprocessing image ${imagePath}`);

    const image = fs.readFileSync(imagePath);
    const buf = Buffer.from(image);
    const uint8array = new Uint8Array(buf);

    return tfnode.node.decodeImage(uint8array, 3).expandDims();
}

// run prediction with the provided input Tensor
const runModel = function (inputTensor) {
    console.log('runnning model');
  
    return model.executeAsync(inputTensor);
}

// determine the classes and max scores from the prediction
const extractClassesAndMaxScores = function (predictionScores) {
  console.log('calculating classes & max scores');

  const scores = predictionScores.dataSync();
  const numBoxesFound = predictionScores.shape[1];
  const numClassesFound = predictionScores.shape[2];

  const maxScores = [];
  const classes = [];

  // for each bounding box returned
  for (let i = 0; i < numBoxesFound; i++) {
    let maxScore = -1;
    let classIndex = -1;

    // find the class with the highest score
    for (let j = 0; j < numClassesFound; j++) {
      if (scores[i * numClassesFound + j] > maxScore) {
        maxScore = scores[i * numClassesFound + j];
        classIndex = j;
      }
    }

    maxScores[i] = maxScore;
    classes[i] = classIndex;
  }

  return [maxScores, classes];
}

let height = 1;
let width = 1;
// create JSON object with bounding boxes and label
const createJSONresponse = function (boxes, scores, indexes, classes) {
    console.log('create JSON output');
  
    const count = indexes.length;
    const objects = [];
  
    for (let i = 0; i < count; i++) {
      const bbox = [];
  
      for (let j = 0; j < 4; j++) {
        bbox[j] = boxes[indexes[i] * 4 + j];
      }
  
      const minY = bbox[0] * height;
      const minX = bbox[1] * width;
      const maxY = bbox[2] * height;
      const maxX = bbox[3] * width;
  
      objects.push({
        bbox: [minX, minY, maxX, maxY],
        label: labels[classes[indexes[i]]],
        score: scores[indexes[i]]
      });
    }
  
    return objects;
  }

// process the model output into a friendly JSON format
const processOutput = function (prediction) {
    console.log('processOutput');
  
    const [maxScores, classes] = extractClassesAndMaxScores(prediction[0]);
    const indexes = calculateNMS(prediction[1], maxScores);
  
    return createJSONresponse(prediction[1].dataSync(), maxScores, indexes, classes);
  }

 
let imagePath = './images/image1.jpg';

loadModel().then(model => {
    const inputTensor = processInput(imagePath);
    height = inputTensor.shape[1];
    width = inputTensor.shape[2];
    return runModel(inputTensor);
}).then(predection => {
    const output = processOutput(predection);
    console.log(output);
});