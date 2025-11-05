let model;
const CLASS_NAMES = [
  "Apple___Apple_scab",
  "Apple___Black_rot", 
  "Apple___Cedar_apple_rust",
  "Apple___healthy",
  "Blueberry___healthy",
  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
  "Corn_(maize)___Common_rust_",
  "Corn_(maize)___Northern_Leaf_Blight", 
  "Corn_(maize)___healthy",
  "Grape___Black_rot",
  "Grape___Esca_(Black_Measles)",
  "Grape___healthy",
];

const video = document.getElementById('camera');
const captureBtn = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const ctx = snapshotCanvas.getContext('2d');

captureBtn.disabled = true;

async function loadModel() {
  try {
    console.log('Loading model from: plant_model_js/model.json');
    model = await tf.loadLayersModel('plant_model_js/model.json');
    captureBtn.disabled = false;
    resultDiv.innerText = 'Model ready! 👉 Take a photo to analyze';
  } catch (error) {
    console.error('Error loading model:', error);
    resultDiv.innerText = 'Error loading model. Please refresh the page.';
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { 
        facingMode: 'environment',
        width: { ideal: 224 },
        height: { ideal: 224 }
      }
    });
    video.srcObject = stream;
    await video.play();
  } catch (error) {
    console.error('Error accessing camera:', error);
    resultDiv.innerText = 'Error accessing camera. Please check permissions.';
  }
}

function captureImage() {
  resultDiv.innerText = 'Analyzing... 🔍';
  captureBtn.disabled = true;
 
  snapshotCanvas.width = video.videoWidth;
  snapshotCanvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);
  
  if (model) {
    predict();
  } else {
    captureBtn.disabled = false;
    resultDiv.innerText = 'Model not loaded. Please refresh.';
  }
}

async function predict() {
  try {
    const prediction = tf.tidy(() => {
      const img = tf.browser.fromPixels(snapshotCanvas)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(tf.scalar(255))
        .expandDims();
      
      return model.predict(img);
    });

    const values = await prediction.data();
    const maxIndex = values.indexOf(Math.max(...values));
    const predictedClass = CLASS_NAMES[maxIndex];
    const confidence = (values[maxIndex] * 100).toFixed(2);

    resultDiv.innerHTML = `
      🌳 <strong>Detected Disease:</strong> ${predictedClass}<br>
      📊 <strong>Confidence:</strong> ${confidence}%
    `;
  } catch (error) {
    console.error('Prediction error:', error);
    resultDiv.innerText = 'Error during analysis. Please try again.';
  } finally {
    captureBtn.disabled = false;
  }
}

captureBtn.addEventListener('click', captureImage);

(async () => {
  await Promise.all([startCamera(), loadModel()]);
})();
