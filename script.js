// =======================
// script.js
// =======================
let model;
const CLASS_NAMES = [
  // ⚠️ Điền đầy đủ tên class đã train (ví dụ trong PlantVillage)
  "Apple___Apple_scab",
  "Apple___Black_rot",
  "Apple___Cedar_apple_rust",
  "Apple___healthy",
  "Corn___Cercospora_leaf_spot",
  "Corn___Common_rust",
  "Corn___Northern_Leaf_Blight",
  "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
];

const video = document.getElementById('camera');
const captureBtn = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const ctx = snapshotCanvas.getContext('2d');

// =======================
// Kết nối camera
// =======================
async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await video.play();
  } catch (err) {
    console.error("Không thể truy cập camera:", err);
    resultDiv.innerText = "⚠️ Không thể truy cập camera!";
  }
}

// =======================
// Load AI model TF.js
// =======================
async function loadModel() {
  resultDiv.innerText = "⌛ Đang load model AI...";
  try {
    model = await tf.loadLayersModel("plant_model_js/model.json");
    resultDiv.innerText = "✅ Model AI đã sẵn sàng!";
    console.log("Model loaded!");
  } catch (err) {
    console.error("Không thể load model:", err);
    resultDiv.innerText = "⚠️ Không thể load model AI! Lỗi: " + err.message;
  }
}

// =======================
// Chụp ảnh và dự đoán
// =======================
captureBtn.addEventListener("click", async () => {
  if (!model) {
    alert("Model chưa load xong!");
    return;
  }

  snapshotCanvas.width = video.videoWidth;
  snapshotCanvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);

  let tensor = tf.browser.fromPixels(snapshotCanvas)
    .resizeNearestNeighbor([224, 224]) // ⚠️ chỉnh theo kích thước train
    .toFloat()
    .expandDims();

  // Chuẩn hóa [-1,1] nếu dùng MobileNetV2
  const offset = tf.scalar(127.5);
  tensor = tensor.sub(offset).div(offset);

  try {
    const predictions = await model.predict(tensor).data();
    const maxIndex = predictions.indexOf(Math.max(...predictions));
    const predictedClass = CLASS_NAMES[maxIndex] || "Unknown";
    const confidence = (predictions[maxIndex] * 100).toFixed(2);

    resultDiv.innerHTML = `
      🌿 Prediction: <b>${predictedClass}</b><br>
      📊 Confidence: ${confidence}%
    `;
  } catch (err) {
    console.error("Lỗi khi dự đoán:", err);
    resultDiv.innerText = "⚠️ Lỗi khi dự đoán!";
  }
});

// =======================
// Khởi tạo
// =======================
setupCamera();
loadModel();
