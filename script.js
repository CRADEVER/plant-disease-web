let model;
const CLASS_NAMES = [
  "Apple___Apple_scab",
  "Apple___Black_rot", 
  "Apple___Cedar_apple_rust",
  "Apple___healthy",
  "Blueberry___healthy",
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
  } catch (error) {
    console.error('Error loading model:', error);
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { 
        facingMode: 'environment',
        width: { ideal: 256 },
        height: { ideal: 256 }
      }
    });
    video.srcObject = stream;
    await video.play();
  } catch (error) {
    console.error('Error accessing camera:', error);
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
  }
}

async function predict() {
  try {
    const prediction = tf.tidy(() => {
      const img = tf.browser.fromPixels(snapshotCanvas)
        .resizeNearestNeighbor([256, 256])
        .toFloat()
        .sub(tf.scalar(127.5)) 
        .div(tf.scalar(127.5))
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
  } finally {
    captureBtn.disabled = false;
  }
}

captureBtn.addEventListener('click', captureImage);

(async () => {
  await Promise.all([startCamera(), loadModel()]);
})();