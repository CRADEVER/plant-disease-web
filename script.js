// =======================
// script.js
// =======================
let model;
const CLASS_NAMES = [
  // ⚠️ Điền đầy đủ class theo dataset khi train
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
const snapshotCanvas = document.getElementById('snapshot'); // Corrected this line
const resultDiv = document.getElementById('result');
const ctx = snapshotCanvas.getContext('2d');

// Load the model
async function loadModel() {
  resultDiv.innerText = 'Đang tải model...';
  console.log('Bắt đầu tải model...'); // Add this log

  try {
    // UPDATE THIS PATH to where you uploaded the plant_model_js folder
    console.log('Loading model from:', 'plant_model_js/model.json'); // Log the model path
    model = await tf.loadLayersModel('plant_model_js/model.json', {
      inputs: tf.layers.input({shape: [224, 224, 3]})
    });
    console.log('Model đã tải xong.'); // Add this log
    resultDiv.innerText = 'Model đã tải xong. Hãy chụp ảnh để phân tích';
    captureBtn.disabled = false;
  } catch (error) {
    console.error('Lỗi khi tải model:', error); // Add this log for errors during loading
    resultDiv.innerText = 'Không thể tải model. Kiểm tra console để biết chi tiết.'; // Update user message
  }
}

// Start camera
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.play();
  } catch (err) {
    console.error("Error accessing camera: ", err);
    resultDiv.innerText = 'Không thể truy cập camera.';
  }
}

// Capture snapshot and predict
captureBtn.addEventListener('click', () => {
  if (!snapshotCanvas) { // Add check for canvas element
    console.error("snapshotCanvas element not found!");
    resultDiv.innerText = 'Lỗi: Không tìm thấy element snapshotCanvas.';
    return;
  }
  snapshotCanvas.width = video.videoWidth;
  snapshotCanvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);

  if (model) {
    predict();
  } else {
    resultDiv.innerText = 'Model chưa tải xong.';
  }
});

// Predict function
async function predict() {
  if (!model) { // Add check if model is loaded before predicting
    resultDiv.innerText = 'Model chưa tải xong.';
    return;
  }
  resultDiv.innerText = 'Đang phân tích...';
  try { // Add try-catch for prediction
    const img = tf.browser.fromPixels(snapshotCanvas).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
    const predictions = model.predict(img);
    const predictedClass = CLASS_NAMES[predictions.argMax(-1).dataSync()[0]];
    const probability = predictions.dataSync()[predictions.argMax(-1).dataSync()[0]];

    resultDiv.innerText = `Kết quả: ${predictedClass} (Độ chính xác: ${Math.round(probability * 100)}%)`;
  } catch (error) {
    console.error('Lỗi khi phân tích ảnh:', error); // Log prediction errors
    resultDiv.innerText = 'Lỗi khi phân tích ảnh. Kiểm tra console.';
  }
}

// Initialize
startCamera();
loadModel();