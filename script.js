// Khai báo các biến DOM element
const video = document.getElementById('camera');
const captureButton = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const fileUploadInput = document.getElementById('file-upload');
const cameraSection = document.getElementById('camera-section'); 
const previewImg = document.getElementById('uploaded-image-preview') || document.createElement('img');
if (!document.getElementById('uploaded-image-preview')) {
    previewImg.id = 'uploaded-image-preview';
    cameraSection.appendChild(previewImg);
    previewImg.style.display = 'none'; // Ẩn ban đầu
}

// Đường dẫn đến mô hình đã chuyển đổi (Cần thay đổi nếu tên thư mục khác)
const MODEL_URL = 'plant_model_js/model.json'; 
let model;
let classNames = window.CLASS_NAMES || []; // Sẽ được load từ labels.js

/**
 * Tải mô hình TF.js và file nhãn (labels).
 */
async function loadModel() {
    resultDiv.textContent = '⏳ Đang tải mô hình AI...';
    try {
        // Tải mô hình
        model = await tf.loadLayersModel(MODEL_URL);
        
        // Load nhãn từ file labels.js (được tạo ở bước 4)
        if (typeof CLASS_NAMES !== 'undefined') {
            classNames = CLASS_NAMES;
        }

        resultDiv.textContent = `✅ Mô hình đã sẵn sàng. (${classNames.length} loại bệnh)`;
    } catch (err) {
        console.error("Lỗi khi tải mô hình:", err);
        resultDiv.textContent = '❌ Không thể tải mô hình AI. Vui lòng kiểm tra đường dẫn.';
    }
}

/**
 * Xử lý hình ảnh trên canvas và chạy dự đoán.
 */
async function runModelPrediction() {
    if (!model) {
        resultDiv.textContent = '⚠️ Mô hình chưa được tải. Đang tải lại...';
        await loadModel();
        if (!model) return;
    }
    
    resultDiv.textContent = '🧠 Đang phân tích hình ảnh...';
    
    // 1. Chuyển đổi Canvas thành Tensor
    const IMAGE_SIZE = 256; // Kích thước ảnh đã train (xem plant_model_tfjs.ipynb)
    
    let tensor = tf.browser.fromPixels(snapshotCanvas)
        .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE]) // Resize về kích thước mong muốn
        .toFloat();

    // 2. Tiền xử lý (chuẩn hóa MobileNetV2)
    // MobileNetV2 cần giá trị pixel trong khoảng [-1, 1].
    tensor = tf.sub(tf.div(tensor, 127.5), 1);
    
    // 3. Thêm chiều batch (1, 256, 256, 3)
    const expandedTensor = tensor.expandDims(0);
    
    // 4. Dự đoán
    const prediction = await model.predict(expandedTensor).data();
    
    // 5. Xử lý kết quả
    const predictedClassIndex = prediction.indexOf(Math.max(...prediction));
    const predictedClass = classNames[predictedClassIndex] || 'Không xác định';
    const confidence = prediction[predictedClassIndex] * 100;

    // 6. Hiển thị kết quả
    let resultText = `Kết quả: **${predictedClass}**`;
    
    // Thêm gợi ý đơn giản dựa trên kết quả
    if (predictedClass.includes('healthy')) {
        resultText += ' (Cây khỏe mạnh) 🎉';
    } else if (predictedClass !== 'Không xác định') {
        resultText += ' (Cây bị bệnh!) 🚨';
    }
    
    resultDiv.innerHTML = `**Độ tin cậy:** ${confidence.toFixed(2)}%<br>${resultText}`;

    // Giải phóng bộ nhớ Tensor
    tf.dispose([tensor, expandedTensor]);
}


/**
 * Khởi tạo camera và model khi trang load.
 */
async function startCamera() {
    // Luôn tải model trước
    await loadModel();

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        
        // Chỉ cập nhật trạng thái nếu model đã load thành công
        if (model) {
            resultDiv.textContent = `✅ Camera và mô hình đã sẵn sàng.`;
        } else {
             resultDiv.textContent = `❌ Lỗi mô hình. Camera đã sẵn sàng.`;
        }

    } catch (err) {
        console.error("Lỗi khi truy cập camera:", err);
        resultDiv.textContent = '❌ Không thể truy cập camera. Vui lòng kiểm tra quyền.';
        video.style.display = 'none'; 
    }
}

/**
 * Chụp ảnh và chuyển canvas
 */
captureButton.addEventListener('click', () => {
    if (video.srcObject) {
        // 1. Vẽ lên Canvas
        snapshotCanvas.width = video.videoWidth;
        snapshotCanvas.height = video.videoHeight;
        const context = snapshotCanvas.getContext('2d');
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        
        // 2. Hiển thị ảnh đã chụp
        const imageDataURL = snapshotCanvas.toDataURL('image/jpeg');
        video.style.display = 'none';
        previewImg.src = imageDataURL;
        previewImg.style.display = 'block';

        // 3. Phân tích
        resultDiv.textContent = '📸 Ảnh đã chụp. Sẵn sàng phân tích.';
        runModelPrediction();

    } else {
        resultDiv.textContent = '⚠️ Camera chưa sẵn sàng hoặc đã bị tắt.';
    }
});

/**
 * Tải file ảnh lên
 */
fileUploadInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            video.style.display = 'none';
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            
            // 1. Load ảnh vào Canvas để chuẩn bị phân tích
            const img = new Image();
            img.onload = function() {
                snapshotCanvas.width = img.width;
                snapshotCanvas.height = img.height;
                const context = snapshotCanvas.getContext('2d');
                context.drawImage(img, 0, 0);
                
                // 2. Phân tích
                runModelPrediction();
            };
            img.src = e.target.result;

            resultDiv.textContent = `⬆️ Đã tải lên "${file.name}". Sẵn sàng phân tích.`;
        };
        
        reader.readAsDataURL(file);
    }
});


window.onload = startCamera;