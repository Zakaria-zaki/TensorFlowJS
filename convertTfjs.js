import { loadModel } from 'tfjs-onnx';

var modelUrl = 'model.onnx';

// Initialize the tf.model
var model = new loadModel(modelUrl);

// Now use tf.model
const pixels = tf.fromPixels(img);
const predictions = model.predict(pixels);

console.log(predictions);