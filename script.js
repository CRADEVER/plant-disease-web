const video = document.getElementById('camera');
const captureButton = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const fileUploadInput = document.getElementById('file-upload');
const cameraSection = document.getElementById('camera-section');

let model = null;
let CLASS_NAMES = [];
const IMG_SIZE = 256; 

// ĐÃ SỬA: Đường dẫn tuyệt đối, sử dụng thư mục plant_model_js
const MODEL_PATH = '/plant-disease-web/plant_model_js/model.json';
// Đường dẫn tuyệt đối tới file tên lớp (class_indices.json)
const CLASS_INDICES_PATH = '/plant-disease-web/class_indices.json'; 

// =========================
// Khởi động Camera
// =========================
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        resultDiv.textContent = '✅ Camera đã sẵn sàng. Đang tải model...';
    } catch (err) {
        console.error("Lỗi khi truy cập camera:", err);
        resultDiv.textContent = '❌ Không thể truy cập camera. Vui lòng kiểm tra quyền.';
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
        
        // Chuyển đổi JSON object thành mảng
        CLASS_NAMES = Object.values(data);
        
        // 2. Tải Model
        model = await tf.loadLayersModel(MODEL_PATH); // Sử dụng loadLayersModel
        
        resultDiv.innerHTML = '✅ Model và tên lớp đã tải thành công! Bắt đầu chụp ảnh hoặc tải ảnh lên.';
    } catch (error) {
        console.error("Lỗi khi tải tài nguyên:", error);
        resultDiv.innerHTML = `❌ Lỗi: Không thể tải model hoặc tên lớp. Vui lòng kiểm tra đường dẫn hoặc file model trên GitHub.`;
    }
}

// =========================
// Tiền xử lý Tensor và Nhận diện
// =========================
async function predictImage(canvas) {
    if (model === null) {
        resultDiv.textContent = "Model chưa sẵn sàng. Vui lòng đợi.";
        return;
    }

    resultDiv.textContent = "⏳ Đang xử lý và phân tích ảnh...";

    try {
        // 1. Tiền xử lý ảnh (Chuẩn hóa về [0, 1] cho Keras Layers Model)
        const tensor = tf.browser.fromPixels(canvas)
            .resizeBilinear([IMG_SIZE, IMG_SIZE])
            .toFloat()
            .div(tf.scalar(255)) 
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
        // (Logic chụp ảnh và ẩn/hiện đã được đơn giản hóa)
        const previewImg = document.getElementById('uploaded-image-preview');
        previewImg.style.display = 'none';

        snapshotCanvas.width = video.videoWidth;
        snapshotCanvas.height = video.videoHeight;
        const context = snapshotCanvas.getContext('2d');
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        
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
            let previewImg = document.getElementById('uploaded-image-preview');
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';

            const img = new Image();
            img.onload = function() {
                snapshotCanvas.width = img.width;
                snapshotCanvas.height = img.height;
                const context = snapshotCanvas.getContext('2d');
                context.drawImage(img, 0, 0);
                
                video.style.display = 'none';
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