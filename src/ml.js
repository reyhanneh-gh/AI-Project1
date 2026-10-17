import * as tf from "@tensorflow/tfjs";

export const GESTURES = ["Open palm", "Fist", "Thumbs up", "Peace sign"];
export const EXPRESSIONS = ["Neutral", "Happy", "Sad", "Surprised"];
export const IMAGE_SIZE = 64;

export async function initializeTensorflow() {
  await tf.setBackend("webgl");
  await tf.ready();
  return tf.getBackend();
}

export function cropFromVideo(video, box, padding = 0.18) {
  if (!video || !box || !video.videoWidth || !video.videoHeight) return null;
  const x1 = Math.max(0, Math.floor(box.x - box.w * padding));
  const y1 = Math.max(0, Math.floor(box.y - box.h * padding));
  const x2 = Math.min(video.videoWidth, Math.ceil(box.x + box.w * (1 + padding)));
  const y2 = Math.min(video.videoHeight, Math.ceil(box.y + box.h * (1 + padding)));
  const w = x2 - x1, h = y2 - y1;
  if (w < 12 || h < 12) return null;
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, x1, y1, w, h, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  return canvas;
}



export function imageCanvasToTensor(canvas) {
  return tf.tidy(() =>
    tf.browser.fromPixels(canvas)
      .toFloat()
      .div(127.5)
      .sub(1)
  );
}




function buildSharedCnn() {
  console.log("CNN 1: Creating input...");
  const input = tf.input({ shape: [IMAGE_SIZE, IMAGE_SIZE, 3] });

  console.log("CNN 2: First convolution...");
  let x = tf.layers.conv2d({
    filters: 16, kernelSize: 3, strides: 2,
    padding: "same", activation: "relu"
  }).apply(input);

  console.log("CNN 3: First pooling...");
  x = tf.layers.maxPooling2d({ poolSize: 2 }).apply(x);

  console.log("CNN 4: Second convolution...");
  x = tf.layers.conv2d({
    filters: 32, kernelSize: 3,
    padding: "same", activation: "relu"
  }).apply(x);

  console.log("CNN 5: Second pooling...");
  x = tf.layers.maxPooling2d({ poolSize: 2 }).apply(x);

  console.log("CNN 6: Third convolution...");
  x = tf.layers.conv2d({
    filters: 64, kernelSize: 3,
    padding: "same", activation: "relu"
  }).apply(x);

  console.log("CNN 7: Global average pooling...");
  x = tf.layers.globalAveragePooling2d({
    dataFormat: "channelsLast"
  }).apply(x);

  console.log("CNN 8: Dense layer...");
  x = tf.layers.dense({ units: 64, activation: "relu" }).apply(x);

  console.log("CNN 9: Building shared model...");
  const model = tf.model({
    inputs: input,
    outputs: x,
    name: "shared_cnn"
  });

  console.log("CNN 10: Shared model created!");
  return model;
}


function attachHeads(features, prefix) {
  const hidden = tf.layers.dense({ units: 64, activation: "relu", name: `${prefix}_shared_dense` }).apply(features);
  const gesture = tf.layers.dense({ units: GESTURES.length, activation: "softmax", name: "gesture" }).apply(hidden);
  const expression = tf.layers.dense({ units: EXPRESSIONS.length, activation: "softmax", name: "expression" }).apply(hidden);
  return [gesture, expression];
}

export function createCnnBaseline() {
  const faceInput = tf.input({ shape: [IMAGE_SIZE, IMAGE_SIZE, 3], name: "face_image" });
  const handInput = tf.input({ shape: [IMAGE_SIZE, IMAGE_SIZE, 3], name: "hand_image" });
  const cnn = buildSharedCnn();
  const faceFeatures = cnn.apply(faceInput);
  const handFeatures = cnn.apply(handInput);
  const joined = tf.layers.concatenate().apply([faceFeatures, handFeatures]);
  const outputs = attachHeads(joined, "baseline");
  const model = tf.model({ inputs: [faceInput, handInput], outputs, name: "cnn_baseline" });
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: ["categoricalCrossentropy", "categoricalCrossentropy"],
    metrics: ["accuracy"]
  });
  return model;
}

export function createAttentionModel() {
  const faceInput = tf.input({ shape: [IMAGE_SIZE, IMAGE_SIZE, 3], name: "face_image" });
  const handInput = tf.input({ shape: [IMAGE_SIZE, IMAGE_SIZE, 3], name: "hand_image" });
  const cnn = buildSharedCnn();
  const faceFeatures = cnn.apply(faceInput);
  const handFeatures = cnn.apply(handInput);

  // Separate projections tell the attention block which token came from which region.
  const faceToken = tf.layers.dense({ units: 64, activation: "linear", name: "face_token_projection" }).apply(faceFeatures);
  const handToken = tf.layers.dense({ units: 64, activation: "linear", name: "hand_token_projection" }).apply(handFeatures);
  const faceSeq = tf.layers.reshape({ targetShape: [1, 64] }).apply(faceToken);
  const handSeq = tf.layers.reshape({ targetShape: [1, 64] }).apply(handToken);
  const tokens = tf.layers.concatenate({ axis: 1 }).apply([faceSeq, handSeq]); // [batch, 2, 64]

  const q = tf.layers.dense({ units: 64, useBias: false, name: "query" }).apply(tokens);
  const k = tf.layers.dense({ units: 64, useBias: false, name: "key" }).apply(tokens);
  const v = tf.layers.dense({ units: 64, useBias: false, name: "value" }).apply(tokens);
  const scores = tf.layers.dot({ axes: [2, 2], name: "attention_scores" }).apply([q, k]); // [batch, 2, 2]
  const weights = tf.layers.softmax({ axis: -1, name: "attention_weights" }).apply(scores);
  const attended = tf.layers.dot({ axes: [2, 1], name: "attention_output" }).apply([weights, v]);
  const residual = tf.layers.add().apply([tokens, attended]);
  const norm = tf.layers.layerNormalization({ axis: -1, name: "attention_norm" }).apply(residual);
  const ff1 = tf.layers.dense({ units: 128, activation: "relu", name: "ffn_1" }).apply(norm);
  const ff2 = tf.layers.dense({ units: 64, activation: "linear", name: "ffn_2" }).apply(ff1);
  const ffResidual = tf.layers.add().apply([norm, ff2]);
  const ffNorm = tf.layers.layerNormalization({ axis: -1, name: "ffn_norm" }).apply(ffResidual);
  const flattened = tf.layers.flatten().apply(ffNorm);
  const outputs = attachHeads(flattened, "attention");
  const model = tf.model({ inputs: [faceInput, handInput], outputs, name: "cnn_attention_fusion" });
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: ["categoricalCrossentropy", "categoricalCrossentropy"],
    metrics: ["accuracy"]
  });
  return model;
}

export function oneHot(index, count) {
  const a = new Array(count).fill(0);
  a[index] = 1;
  return a;
}

export function makeDataset(samples) {
  const faceTensors = [];
  const handTensors = [];
  try {
    for (const s of samples) {
      faceTensors.push(imageCanvasToTensor(s.face));
      handTensors.push(imageCanvasToTensor(s.hand));
    }
    return {
      xs: [tf.stack(faceTensors), tf.stack(handTensors)],
      ys: [
        tf.tensor2d(samples.map(s => oneHot(s.gesture, GESTURES.length))),
        tf.tensor2d(samples.map(s => oneHot(s.expression, EXPRESSIONS.length)))
      ]
    };
  } finally {
    faceTensors.forEach(t => t.dispose());
    handTensors.forEach(t => t.dispose());
  }
}

export function evaluatePredictions(model, samples) {
  const ds = makeDataset(samples);
  try {
    const preds = model.predict(ds.xs);
    const gesturePred = preds[0].argMax(-1).dataSync();
    const expressionPred = preds[1].argMax(-1).dataSync();
    const gestureTrue = ds.ys[0].argMax(-1).dataSync();
    const expressionTrue = ds.ys[1].argMax(-1).dataSync();
    let gCorrect = 0, eCorrect = 0, joint = 0;
    for (let i = 0; i < samples.length; i++) {
      if (gesturePred[i] === gestureTrue[i]) gCorrect++;
      if (expressionPred[i] === expressionTrue[i]) eCorrect++;
      if (gesturePred[i] === gestureTrue[i] && expressionPred[i] === expressionTrue[i]) joint++;
    }
    tf.dispose(preds);
    return {
      gesture: gCorrect / samples.length,
      expression: eCorrect / samples.length,
      joint: joint / samples.length
    };
  } finally {
    tf.dispose([...ds.xs, ...ds.ys]);
  }
}

export function predictSample(model, faceCanvas, handCanvas) {
  const face = imageCanvasToTensor(faceCanvas).expandDims(0);
  const hand = imageCanvasToTensor(handCanvas).expandDims(0);
  try {
    const outputs = model.predict([face, hand]);
    const gesture = Array.from(outputs[0].dataSync());
    const expression = Array.from(outputs[1].dataSync());
    tf.dispose(outputs);
    return { gesture, expression };
  } finally {
    face.dispose();
    hand.dispose();
  }
}
