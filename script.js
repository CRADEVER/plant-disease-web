const video = document.getElementById('camera');
const captureButton = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const fileUploadInput = document.getElementById('file-upload');
const cameraSection = document.getElementById('camera-section');

// --- 0. KHAI BÁO BIẾN TOÀN CỤC CHO AI ---
let model;
const MODEL_URL = 'plant_model_js/model.json'; // Đường dẫn đến model.json
// Tên các lớp (class names) được lấy từ output của plant_model_tfjs.ipynb
const CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy']; //

// 1. Khởi động Camera
async function startCamera() {
    try {
        // Yêu cầu quyền truy cập camera, chỉ lấy video
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        resultDiv.textContent = '✅ Camera đã sẵn sàng. Đang tải model AI...'; // Cập nhật thông báo
        await loadModel(); // Tải model ngay sau khi camera sẵn sàng
    } catch (err) {
        console.error("Lỗi khi truy cập camera:", err);
        resultDiv.textContent = '❌ Không thể truy cập camera. Vui lòng kiểm tra quyền. Đang tải model AI...';
        // Ẩn video nếu không thể khởi động
        video.style.display = 'none'; 
        await loadModel(); // Vẫn tải model để có thể dùng tính năng tải ảnh lên
    }
}

// --- 4. TẢI MODEL AI (TF.js) ---
async function loadModel() {
    try {
        model = await tf.loadLayersModel(MODEL_URL);
        // Lấy kích thước đầu vào (ví dụ: 256) từ shape của model (shape: [null, 256, 256, 3])
        const [_, size, __, ___] = model.inputs[0].shape; 
        
        // Cập nhật canvas để khớp với kích thước input của model
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
        // Lấy kích thước chuẩn từ canvas đã được đặt theo kích thước model
        const size = snapshotCanvas.width;
        
        // Chuyển ảnh từ canvas sang Tensor
        let tensor = tf.browser.fromPixels(snapshotCanvas);

        // Tiền xử lý: Resize, chuẩn hoá về [-1, 1], và thêm chiều batch
        tensor = tf.image.resizeBilinear(tensor, [size, size])
            .toFloat()
            // Chuẩn hoá theo công thức MobileNetV2: (x / 127.5) - 1.0
            .div(127.5)
            .sub(1.0)
            .expandDims(); // Thêm chiều Batch (shape: [1, size, size, 3])
        
        // Thực hiện dự đoán
        const predictions = await model.predict(tensor).data();
        
        // Lấy kết quả có xác suất cao nhất
        const maxProbability = Math.max(...predictions);
        const maxIndex = predictions.indexOf(maxProbability);
        const className = CLASS_NAMES[maxIndex];

        // Format kết quả
        const confidence = (maxProbability * 100).toFixed(2);
        
        // Hiển thị kết quả đơn giản: Cây khỏe mạnh hay bị bệnh gì
        let simpleResult;
        if (className.endsWith('healthy')) {
            simpleResult = `💚 Cây **khỏe mạnh**!`;
        } else {
            // Định dạng lại tên lớp (Ví dụ: Apple___Black_rot -> Apple: Black rot)
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
    // Đảm bảo video đang hiển thị và có luồng dữ liệu
    if (video.srcObject) {
        
        // Vẽ khung hình hiện tại của video lên canvas
        const context = snapshotCanvas.getContext('2d');
        // Kích thước canvas đã được đặt trong loadModel(), ta chỉ cần vẽ lên
        context.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);
        
        // Tạo một thẻ <img> để hiển thị ảnh chụp (giữ nguyên layout)
        let previewImg = document.getElementById('uploaded-image-preview');
        if (!previewImg) {
            previewImg = document.createElement('img');
            previewImg.id = 'uploaded-image-preview';
            cameraSection.appendChild(previewImg);
        }
        previewImg.src = snapshotCanvas.toDataURL('image/jpeg');
        
        // Tạm ẩn video và hiện ảnh xem trước
        video.style.display = 'none';
        previewImg.style.display = 'block';

        resultDiv.textContent = '📸 Ảnh đã chụp. Đang phân tích...';
        
        // --- GỌI HÀM PHÂN TÍCH ---
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
                // Vẽ ảnh lên canvas, resize nó để khớp với kích thước input của model
                context.drawImage(img, 0, 0, snapshotCanvas.width, snapshotCanvas.height); 
                
                resultDiv.textContent = `⬆️ Đã tải lên "${file.name}". Đang phân tích...`;
                
                // --- GỌI HÀM PHÂN TÍCH ---
                analyzeImage();
            };
            img.src = e.target.result;
        };
        
        reader.readAsDataURL(file);
    }
});

// Bắt đầu camera và tải model khi trang web được tải
window.onload = startCamera;