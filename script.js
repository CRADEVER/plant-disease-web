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

// Load the model
async function loadModel() {
  resultDiv.innerText = 'Đang tải model...';
  console.log('Bắt đầu tải model...');

  try {
    console.log('Loading model from:', 'plant_model_js/model.json');
    model = await tf.loadLayersModel('plant_model_js/model.json');
    console.log('✅ Model đã tải xong.');
    resultDiv.innerText = '✅ Model đã tải xong. Hãy chụp ảnh để phân tích';
    captureBtn.disabled = false;
  } catch (error) {
    console.error('❌ Lỗi khi tải model:', error);
    resultDiv.innerText = '⚠️ Không thể tải model. Kiểm tra console để biết chi tiết.';
  }
}

// Start camera
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    console.error("❌ Error accessing camera: ", err);
    resultDiv.innerText = '⚠️ Không thể truy cập camera.';
  }
}

// Capture snapshot and predict
captureBtn.addEventListener('click', () => {
  snapshotCanvas.width = video.videoWidth;
  snapshotCanvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);

  if (model) {
    predict();
  } else {
    resultDiv.innerText = '⚠️ Model chưa tải xong.';
  }
});

// Predict function
async function predict() {
  try {
    const img = tf.browser.fromPixels(snapshotCanvas)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .expandDims();

    const predictions = await model.predict(img).data();
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const predictedClass = CLASS_NAMES[maxIndex];
    const confidence = (predictions[maxIndex] * 100).toFixed(2);

    resultDiv.innerText = `🌿 Kết quả: ${predictedClass}\n📊 Độ chính xác: ${confidence}%`;
  } catch (error) {
    console.error('❌ Lỗi khi phân tích ảnh:', error);
    resultDiv.innerText = '⚠️ Lỗi khi phân tích ảnh. Kiểm tra console.';
  }
}

// Initialize
startCamera();
loadModel();
