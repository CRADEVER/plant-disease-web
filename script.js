const video = document.getElementById('camera');
const captureBtn = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');

const ctx = snapshotCanvas.getContext('2d');

let isCaptured = false; // trạng thái: đã chụp hay chưa

// =========================
// Khởi động camera
// =========================
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
    console.error("Lỗi camera:", error);
  }
}

// =========================
// Chụp ảnh và gửi backend
// =========================
async function captureImage() {
  if (!isCaptured) {
    // ======= Chế độ CHỤP ẢNH =======
    resultDiv.innerText = "⏳ Đang xử lý ảnh...";

    snapshotCanvas.width = video.videoWidth;
    snapshotCanvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);

    // Ẩn camera ➜ hiện ảnh chụp
    video.style.display = "none";
    snapshotCanvas.style.display = "block";

    captureBtn.innerText = "🔄 Tiếp tục";
    isCaptured = true;

    // Gửi ảnh lên backend
    sendToBackend();
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
// Gửi ảnh cho backend Flask
// =========================
async function sendToBackend() {
  snapshotCanvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append("image", blob, "capture.jpg");

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      resultDiv.innerHTML = `
        🌳 <strong>Bệnh phát hiện:</strong> ${data.class}<br>
        📊 <strong>Độ chính xác:</strong> ${(data.confidence * 100).toFixed(2)}%
      `;

    } catch (error) {
      console.error("Lỗi gửi ảnh:", error);
      resultDiv.innerText = "❌ Lỗi khi kết nối backend!";
    }
  }, "image/jpeg");
}

captureBtn.addEventListener("click", captureImage);

// Khởi động camera khi mở trang
startCamera();
resultDiv.innerText = "👉 Hãy chụp ảnh để phân tích";