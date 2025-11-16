const video = document.getElementById('camera');
const captureBtn = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');

const ctx = snapshotCanvas.getContext('2d');

let isCaptured = false; // trạng thái: đã chụp hay chưa
let model = null;
const MODEL_PATH = './plant_model_js/model.json';
const IMG_SIZE = 256; // Phải khớp với lúc train (Cell 4 trong Colab)

// =========================
// Tải Model TensorFlow.js
// =========================
async function loadModel() {
  try {
    model = await tf.loadGraphModel(MODEL_PATH);
    console.log("Model đã tải thành công!");
    resultDiv.innerText = "👉 Hãy chụp ảnh để phân tích";
  } catch (error) {
    console.error("Lỗi khi tải model:", error);
    resultDiv.innerText = "❌ Lỗi: Không thể tải model.";
  }
}

// =========================
// Khởi động camera
// =========================
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'environment',
        width: { ideal: 400 }, // Yêu cầu camera
        height: { ideal: 400 }
      }
    });

    video.srcObject = stream;
    await video.play();

  } catch (error) {
    console.error("Lỗi camera:", error);
    resultDiv.innerText = "❌ Lỗi: Không thể bật camera.";
  }
}

// =========================
// Chụp ảnh và chạy nhận diện
// =========================
async function captureAndPredict() {
  if (model === null) {
    resultDiv.innerText = "Chưa tải xong model, vui lòng đợi...";
    return;
  }

  if (!isCaptured) {
    // ======= Chế độ CHỤP ẢNH =======
    resultDiv.innerText = "⏳ Đang xử lý ảnh...";

    // Đảm bảo kích thước canvas khớp để vẽ
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    snapshotCanvas.width = videoWidth;
    snapshotCanvas.height = videoHeight;
    
    // Vẽ ảnh từ video lên canvas
    // ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
    
    // Căn ảnh vào giữa canvas (quan trọng)
    const shorterSide = Math.min(videoWidth, videoHeight);
    const xOffset = (videoWidth - shorterSide) / 2;
    const yOffset = (videoHeight - shorterSide) / 2;
    ctx.drawImage(video, xOffset, yOffset, shorterSide, shorterSide, 0, 0, snapshotCanvas.width, snapshotCanvas.height);


    // Ẩn camera ➜ hiện ảnh chụp
    video.style.display = "none";
    snapshotCanvas.style.display = "block";

    captureBtn.innerText = "🔄 Tiếp tục";
    isCaptured = true;

    // Gửi ảnh đi nhận diện (NGAY TRONG TRÌNH DUYỆT)
    predictImage(snapshotCanvas);

  } else {
    // ======= Chế độ TIẾP TỤC =======
    resultDiv.innerText = "👉 Hãy chụp ảnh để phân tích";

    // Hiện lại camera ➜ ẩn ảnh chụp
    video.style.display = "block";
    snapshotCanvas.style.display = "none";

    captureBtn.innerText = "📸 Chụp ảnh";
    isCaptured = false;
  }
}

// =========================
// Nhận diện ảnh (Client-side)
// =========================
async function predictImage(canvas) {
  try {
    // 1. Tiền xử lý ảnh
    const tensor = tf.browser.fromPixels(canvas)
      .resizeBilinear([IMG_SIZE, IMG_SIZE]) // Resize về 256x256
      .toFloat()
      .div(tf.scalar(127.5)) // Chuẩn hóa về [-1, 1]
      .sub(tf.scalar(1.0))
      .expandDims(0); // Thêm chiều batch (1, 256, 256, 3)

    // 2. Chạy model
    const predictions = await model.predict(tensor).data();
    tensor.dispose(); // Giải phóng bộ nhớ

    // 3. Tìm kết quả tốt nhất
    let maxProb = -1;
    let maxIndex = -1;

    for (let i = 0; i < predictions.length; i++) {
      if (predictions[i] > maxProb) {
        maxProb = predictions[i];
        maxIndex = i;
      }
    }

    // CLASS_NAMES được tải từ file labels.js
    const predictedClass = CLASS_NAMES[maxIndex];

    // 4. Hiển thị kết quả
    resultDiv.innerHTML = `
      🌳 <strong>Bệnh phát hiện:</strong> ${predictedClass.replace(/_/g, ' ')}<br>
      📊 <strong>Độ chính xác:</strong> ${(maxProb * 100).toFixed(2)}%
    `;

  } catch (error) {
    console.error("Lỗi khi nhận diện:", error);
    resultDiv.innerText = "❌ Lỗi khi phân tích ảnh!";
  }
}

// =========================
// Khởi chạy
// =========================
captureBtn.addEventListener("click", captureAndPredict);

// Bắt đầu tải model và camera ngay khi mở trang
loadModel();
startCamera();
resultDiv.innerText = "⏳ Đang tải model...";