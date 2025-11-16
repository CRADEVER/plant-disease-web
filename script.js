const video = document.getElementById('camera');
const captureButton = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const fileUploadInput = document.getElementById('file-upload');
const cameraSection = document.getElementById('camera-section');
const previewImg = document.getElementById('uploaded-image-preview'); // Thẻ <img> để preview

let model = null;
let CLASS_NAMES = []; // Mảng chứa tên các lớp (labels)
const IMG_SIZE = 256; 

// ĐÃ SỬA: Đường dẫn tuyệt đối đến model và class_indices.json
const MODEL_PATH = '/tên_repo/plant_model_js/model.json';
const CLASS_INDICES_PATH = '/tên_repo/class_indices.json'; 

// =========================
// Khởi động Camera
// =========================
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } // Ưu tiên camera sau
        });
        video.srcObject = stream;
    } catch (err) {
        console.error("Lỗi khi truy cập camera:", err);
        // Ẩn video nếu không bật được camera
        video.style.display = 'none'; 
    }
}

// =========================
// Tải Model và Tên Lớp (Labels)
// =========================
async function initialize() {
    resultDiv.textContent = '⏳ Đang tải model và tên lớp...';
    try {
        // 1. Tải tên lớp từ class_indices.json
        const response = await fetch(CLASS_INDICES_PATH);
        const data = await response.json();
        
        // Chuyển đổi JSON object (0: 'Apple_scab') thành mảng ['Apple_scab', ...]
        CLASS_NAMES = Object.values(data);
        
        // 2. Tải Model (Dùng loadGraphModel cho SavedModel/Keras Model)
        // Nếu bạn gặp lỗi 'producer', thử đổi thành tf.loadLayersModel(MODEL_PATH)
        model = await tf.loadGraphModel(MODEL_PATH);
        
        resultDiv.innerHTML = '✅ Model đã sẵn sàng! Chụp ảnh hoặc tải ảnh lên.';
    } catch (error) {
        console.error("Lỗi khi tải tài nguyên:", error);
        resultDiv.innerHTML = `❌ Lỗi: Không thể tải model hoặc tên lớp. Kiểm tra **Network Tab (F12)**.<br>
            Lỗi phổ biến: **404 Not Found** cho file model.json hoặc các file .bin.`;
    }
}

// =========================
// Nhận diện ảnh (Client-side)
// =========================
async function predictImage(canvas) {
    if (model === null) {
        resultDiv.textContent = "Model chưa sẵn sàng. Vui lòng đợi.";
        return;
    }

    resultDiv.textContent = "⏳ Đang xử lý và phân tích ảnh...";

    try {
        // 1. Tiền xử lý ảnh (MobileNetV2 Preprocessing: [-1, 1])
        const tensor = tf.browser.fromPixels(canvas)
            .resizeBilinear([IMG_SIZE, IMG_SIZE])
            .toFloat()
            .div(tf.scalar(127.5)) 
            .sub(tf.scalar(1.0))
            .expandDims(0); 

        // 2. Chạy model
        const predictions = model.predict(tensor);
        const values = await predictions.data();
        tensor.dispose(); 
        predictions.dispose();

        // 3. Tìm kết quả tốt nhất
        let maxProb = -1;
        let maxIndex = -1;

        for (let i = 0; i < values.length; i++) {
            if (values[i] > maxProb) {
                maxProb = values[i];
                maxIndex = i;
            }
        }

        // 4. Hiển thị kết quả
        if (CLASS_NAMES.length > maxIndex) {
            const predictedClass = CLASS_NAMES[maxIndex];
            resultDiv.innerHTML = `
              🌳 <strong>Bệnh phát hiện:</strong> ${predictedClass.replace(/_/g, ' ')}<br>
              📊 <strong>Độ chính xác:</strong> ${(maxProb * 100).toFixed(2)}%
            `;
        } else {
            resultDiv.innerText = "❌ Lỗi: Không tìm thấy tên lớp. Kiểm tra class_indices.json.";
        }

    } catch (error) {
        console.error("Lỗi khi nhận diện:", error);
        resultDiv.innerText = "❌ Lỗi khi phân tích ảnh!";
    }
}

// =========================
// Xử lý sự kiện CHỤP ẢNH
// =========================
captureButton.addEventListener('click', () => {
    if (video.srcObject) {
        // Tắt hiển thị ảnh đã tải lên
        previewImg.style.display = 'none';

        // Vẽ ảnh từ video vào canvas
        snapshotCanvas.width = video.videoWidth;
        snapshotCanvas.height = video.videoHeight;
        const context = snapshotCanvas.getContext('2d');
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        
        // Ẩn camera, hiện canvas (ảnh chụp)
        video.style.display = 'none';
        snapshotCanvas.style.display = 'block';

        predictImage(snapshotCanvas);
    } else {
        resultDiv.textContent = '⚠️ Camera chưa sẵn sàng.';
    }
});

// =========================
// Xử lý sự kiện TẢI ẢNH LÊN
// =========================
fileUploadInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            
            // Hiển thị ảnh trong thẻ previewImg
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            snapshotCanvas.style.display = 'block'; // Vẫn cần hiện canvas để chạy predict

            const img = new Image();
            img.onload = function() {
                // Vẽ ảnh vào Canvas để tiền xử lý
                snapshotCanvas.width = img.width;
                snapshotCanvas.height = img.height;
                const context = snapshotCanvas.getContext('2d');
                context.drawImage(img, 0, 0);
                
                // Ẩn video camera
                video.style.display = 'none';
                
                // Chạy nhận diện
                predictImage(snapshotCanvas);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

// Khởi chạy khi trang tải xong
startCamera();
initialize();