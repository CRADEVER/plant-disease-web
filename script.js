const video = document.getElementById('camera');
const captureButton = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const fileUploadInput = document.getElementById('file-upload');
const cameraSection = document.getElementById('camera-section');

// --- 0. KHAI BÁO BIẾN TOÀN CỤC CHO AI VÀ KÍCH THƯỚC ---
let model;
const MODEL_URL = 'plant_model_js/model.json';
// Kích thước cố định của ảnh đầu vào (Lấy từ Colab: IMG_SIZE = 256)
const IMG_SIZE = 256; 
// Tên các lớp (class names) theo thứ tự dự đoán của model
const CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy']; 


// 1. Khởi động Camera
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        resultDiv.textContent = '✅ Camera đã sẵn sàng. Đang tải model AI...'; 
        await loadModel(); // Tải model ngay sau khi camera sẵn sàng
    } catch (err) {
        console.error("Lỗi khi truy cập camera:", err);
        resultDiv.textContent = '❌ Không thể truy cập camera. Vui lòng kiểm tra quyền. Đang tải model AI...';
        video.style.display = 'none'; 
        await loadModel(); // Vẫn tải model để có thể dùng tính năng tải ảnh lên
    }
}

// --- 4. TẢI MODEL AI (TF.js) ---
async function loadModel() {
    try {
        model = await tf.loadLayersModel(MODEL_URL);
        
        // SỬ DỤNG KÍCH THƯỚC CỐ ĐỊNH để tránh lỗi đọc shape từ model.json
        const size = IMG_SIZE;
        
        // Cập nhật canvas để khớp với kích thước input của model (256x256)
        snapshotCanvas.width = size;
        snapshotCanvas.height = size;

        resultDiv.textContent = '✅ Model AI đã tải thành công. Sẵn sàng phân tích!';
    } catch (e) {
        console.error("Lỗi khi tải model AI:", e);
        resultDiv.textContent = '❌ Lỗi: Không thể tải model AI. Vui lòng kiểm tra đường dẫn và kết nối mạng.';
    }
}


// --- 5. PHÂN TÍCH ẢNH VÀ HIỂN THỊ KẾT QUẢ ---
async function analyzeImage() {
    if (!model) {
        resultDiv.textContent = '⚠️ Model AI chưa được tải. Vui lòng chờ.';
        return;
    }

    try {
        const size = IMG_SIZE;
        
        // Chuyển ảnh từ canvas sang Tensor
        let tensor = tf.browser.fromPixels(snapshotCanvas);

        // Tiền xử lý: Resize, chuẩn hoá về [-1, 1], và thêm chiều batch
        tensor = tf.image.resizeBilinear(tensor, [size, size])
            .toFloat()
            // Chuẩn hoá theo công thức MobileNetV2: (x / 127.5) - 1.0
            .div(127.5)
            .sub(1.0)
            .expandDims(); 
        
        // Thực hiện dự đoán
        const predictions = await model.predict(tensor).data();
        
        // Lấy kết quả có xác suất cao nhất
        const maxProbability = Math.max(...predictions);
        const maxIndex = predictions.indexOf(maxProbability);
        const className = CLASS_NAMES[maxIndex];

        // Format kết quả
        const confidence = (maxProbability * 100).toFixed(2);
        
        // Hiển thị kết quả đơn giản theo yêu cầu
        let simpleResult;
        if (className.endsWith('healthy')) {
            simpleResult = `💚 Cây **khỏe mạnh**!`;
        } else {
            // Định dạng lại tên lớp
            const displayClassName = className.replace(/___/g, ': ').replace(/_/g, ' ');
            simpleResult = `💔 Cây bị bệnh **${displayClassName}**!`;
        }

        resultDiv.innerHTML = `${simpleResult} (Độ tin cậy: ${confidence}%)`;

        // Dọn dẹp Tensor
        tensor.dispose();

    } catch (e) {
        console.error("Lỗi khi phân tích ảnh:", e);
        resultDiv.textContent = '❌ Lỗi trong quá trình phân tích ảnh.';
    }
}


// 2. Xử lý Chụp ảnh từ Camera
captureButton.addEventListener('click', () => {
    if (video.srcObject) {
        
        const context = snapshotCanvas.getContext('2d');
        // Vẽ khung hình hiện tại của video lên canvas
        context.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);
        
        // Tạo/Cập nhật thẻ <img> để hiển thị ảnh chụp
        let previewImg = document.getElementById('uploaded-image-preview');
        if (!previewImg) {
            previewImg = document.createElement('img');
            previewImg.id = 'uploaded-image-preview';
            cameraSection.appendChild(previewImg);
        }
        previewImg.src = snapshotCanvas.toDataURL('image/jpeg');
        
        video.style.display = 'none';
        previewImg.style.display = 'block';

        resultDiv.textContent = '📸 Ảnh đã chụp. Đang phân tích...';
        
        analyzeImage();
    } else {
        resultDiv.textContent = '⚠️ Camera chưa sẵn sàng hoặc đã bị tắt.';
    }
});


// 3. Xử lý Tải ảnh lên
fileUploadInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            
            // Hiển thị ảnh tải lên
            video.style.display = 'none';
            let previewImg = document.getElementById('uploaded-image-preview');
            if (!previewImg) {
                previewImg = document.createElement('img');
                previewImg.id = 'uploaded-image-preview';
                cameraSection.appendChild(previewImg);
            }
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            
            // Vẽ ảnh tải lên lên canvas (chuẩn bị cho phân tích)
            const img = new Image();
            img.onload = function() {
                const context = snapshotCanvas.getContext('2d');
                // Vẽ ảnh lên canvas, resize nó để khớp với kích thước input của model (256x256)
                context.drawImage(img, 0, 0, snapshotCanvas.width, snapshotCanvas.height); 
                
                resultDiv.textContent = `⬆️ Đã tải lên "${file.name}". Đang phân tích...`;
                
                analyzeImage();
            };
            img.src = e.target.result;
        };
        
        reader.readAsDataURL(file);
    }
});

// Bắt đầu camera và tải model khi trang web được tải
window.onload = startCamera;