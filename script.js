// --- Model & App State ---
let model;
// Giữ nguyên CLASS_NAMES để tham chiếu đến output của mô hình
const CLASS_NAMES = [
  "Apple___Apple_scab", // Bệnh
  "Apple___Black_rot", // Bệnh
  "Apple___Cedar_apple_rust", // Bệnh
  "Apple___healthy", // Khỏe
  "Blueberry___healthy", // Khỏe
];

// Định nghĩa các chỉ mục (index) nào trong CLASS_NAMES là 'Healthy'
// Apple___healthy là index 3, Blueberry___healthy là index 4
const HEALTHY_INDICES = [3, 4];

// --- DOM Elements ---
// Thêm vào phần DOM Elements đầu script.js
// ...
const fileUploadInput = document.getElementById('file-upload');
const resetBtn = document.getElementById('reset-camera'); // Đã thêm ID này trong index.html
const ctx = snapshotCanvas.getContext('2d');
// ...

// Thêm logic hiển thị/ẩn nút reset và camera/canvas trong các hàm:

async function startCamera() {
    // ... (Phần logic cũ)
    video.style.display = 'block';
    snapshotCanvas.style.display = 'none';
    resetBtn.style.display = 'none'; // Ẩn nút reset khi đang ở chế độ camera
    // ... (Phần logic cũ)
}

function captureImageFromCamera() {
    if (video.srcObject) {
        // ... (Phần logic cũ)
        
        // Ẩn video, show canvas để giữ lại ảnh chụp
        video.style.display = 'none';
        snapshotCanvas.style.display = 'block';
        
        // Hiển thị nút reset sau khi chụp
        resetBtn.style.display = 'inline-block'; 

        // Run prediction on the canvas image
        // ... (Phần logic cũ)
    } else {
        // ... (Phần logic cũ)
    }
}

// Thêm Event Listener cho nút Reset (cần thêm lại đoạn này nếu bạn xóa nó)
if (resetBtn) {
  resetBtn.addEventListener('click', startCamera);
}
const video = document.getElementById('camera');
const captureBtn = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const fileUploadInput = document.getElementById('file-upload');
// const resetBtn = document.getElementById('reset-camera'); // Xóa hoặc comment resetBtn

const ctx = snapshotCanvas.getContext('2d');

// Vô hiệu hóa nút cho đến khi model được load
captureBtn.disabled = true;
fileUploadInput.disabled = true;

// --- Core Functions ---

/**
 * Loads the TensorFlow.js model
 */
async function loadModel() {
  try {
    console.log('Loading model from: plant_model_js/model.json');
    resultDiv.innerHTML = 'Loading model... 🧠';
    model = await tf.loadLayersModel('plant_model_js/model.json');
    
    // Bật controls khi model đã load
    captureBtn.disabled = false;
    fileUploadInput.disabled = false;
    resultDiv.innerHTML = 'Model loaded. Camera ready. 📸';
  } catch (error) {
    console.error('Error loading model:', error);
    resultDiv.innerHTML = '❌ Error loading model.';
  }
}

/**
 * Starts the camera stream (using rear camera)
 */
async function startCamera() {
  // Logic giữ nguyên
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'environment', // Use rear camera 
        width: { ideal: 256 },
        height: { ideal: 256 }
      }
    });
    video.srcObject = stream;
    await video.play();

    // Show video, hide canvas
    video.style.display = 'block';
    snapshotCanvas.style.display = 'none';
    snapshotCanvas.width = 0; // Đảm bảo canvas bị ẩn hoàn toàn

    // Re-enable capture button if model is loaded
    if (model) {
      captureBtn.disabled = false;
      resultDiv.innerHTML = 'Camera ready. 📸';
    }
  } catch (error) {
    console.error('Error accessing camera:', error);
    resultDiv.innerHTML = '❌ Error accessing camera. Please grant permission.';
  }
}

/**
 * Runs the prediction on the image currently in the canvas
 * CHỈNH SỬA LOGIC HIỂN THỊ KẾT QUẢ TẠI ĐÂY
 */
async function predict() {
  try {
    const prediction = tf.tidy(() => {
      // Get image data from the canvas
      const img = tf.browser.fromPixels(snapshotCanvas)
        .resizeNearestNeighbor([256, 256]) // Model expects 256x256
        .toFloat()
        .sub(tf.scalar(127.5)) // Normalize pixel values
        .div(tf.scalar(127.5))
        .expandDims(); // Add batch dimension

      return model.predict(img);
    });

    const values = await prediction.data();
    const maxIndex = values.indexOf(Math.max(...values));
    const predictedClass = CLASS_NAMES[maxIndex];
    const confidence = (values[maxIndex] * 100).toFixed(2);
    
    // --- LOGIC MỚI: Phân loại thành Healthy/Diseased ---
    let resultText = '';
    let resultClass = ''; // Để dùng cho CSS (tối ưu phần 3)

    if (HEALTHY_INDICES.includes(maxIndex)) {
      // Nếu index dự đoán nằm trong danh sách Healthy_INDICES
      resultText = `✅ Cây **khỏe mạnh** (${predictedClass})`;
      resultClass = 'healthy';
    } else {
      // Nếu không, đó là bệnh
      resultText = `⚠️ Cây **bị bệnh**! (${predictedClass})`;
      resultClass = 'diseased';
    }

    // Hiển thị kết quả đã đơn giản hóa
    resultDiv.className = resultClass; // Gán class để CSS định dạng
    resultDiv.innerHTML = `
      ${resultText}<br>
      📊 Độ tin cậy: ${confidence}%
    `;

  } catch (error) {
    console.error('Prediction error:', error);
    resultDiv.innerHTML = '❌ Error during prediction.';
  } finally {
    // Re-enable capture button (if we're in camera mode)
    if (video.style.display === 'block') {
      captureBtn.disabled = false;
    }
  }
}

// --- Event Listeners ---

/**
 * 1. Capture image from camera feed
 */
function captureImageFromCamera() {
  if (video.srcObject) {
    resultDiv.innerHTML = 'Analyzing... 🔍';
    // Xóa class cũ trước khi phân tích
    resultDiv.className = ''; 
    captureBtn.disabled = true;

    // Draw the current video frame to the (hidden) canvas
    snapshotCanvas.width = video.videoWidth;
    snapshotCanvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);

    // Ẩn video, show canvas để giữ lại ảnh chụp
    video.style.display = 'none';
    snapshotCanvas.style.display = 'block';

    // Run prediction on the canvas image
    if (model) {
      predict();
    }
  } else {
    resultDiv.innerHTML = '⚠️ Camera not started.';
  }
}
captureBtn.addEventListener('click', captureImageFromCamera);

/**
 * 2. Handle file upload
 */
fileUploadInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();

    reader.onload = function(e) {
      const img = new Image();
      img.onload = function() {
        // Hide the video feed
        video.style.display = 'none';
        // Show the canvas and draw the uploaded image on it
        snapshotCanvas.style.display = 'block';
        snapshotCanvas.width = img.width;
        snapshotCanvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        // Clear result div class
        resultDiv.className = ''; 
        // Now that the image is on the canvas, predict
        resultDiv.innerHTML = 'Analyzing uploaded image... 🔍';
        captureBtn.disabled = true; // Disable camera button
        if (model) {
          predict();
        }
      };
      img.src = e.target.result;
    };

    reader.readAsDataURL(file);
  }
});

// --- Initial Startup ---
(async () => {
  // Start camera and load model at the same time
  await Promise.all([startCamera(), loadModel()]);
})();