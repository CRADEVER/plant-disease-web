const video = document.getElementById('camera');
const captureButton = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const fileUploadInput = document.getElementById('file-upload');
const cameraSection = document.getElementById('camera-section');

// Kích thước đầu vào của mô hình (256x256 pixel)
const MODEL_SIZE = 256; 
let model;
let classNames = []; 

// === HÀM TẢI MÔ HÌNH VÀ LABELS ===
async function loadModelAndLabels() {
    try {
        // Tải labels từ file labels.js (Bây giờ nó là ES Module)
        const module = await import('./plant_model_js/labels.js');
        classNames = module.CLASS_NAMES;
        console.log(`Đã tải ${classNames.length} classes.`);
        
        // Tải mô hình TF.js
        model = await tf.loadLayersModel('./plant_model_js/model.json');
        console.log("✅ Model TF.js đã tải thành công.");
        
        // Bắt đầu camera sau khi model đã sẵn sàng
        startCamera();
    } catch (error) {
        console.error("Lỗi khi tải model hoặc labels:", error);
        resultDiv.textContent = '❌ Lỗi: Không thể tải mô hình phân tích. Vui lòng kiểm tra thư mục plant_model_js.';
        resultDiv.style.color = '#d32f2f'; // Đặt màu đỏ cho lỗi
        startCamera(); 
    }
}

// === HÀM PHÂN TÍCH ẢNH ===
async function predictImage() {
    if (!model || classNames.length === 0) {
        resultDiv.textContent = '⚠️ Mô hình chưa sẵn sàng. Đang chờ tải...';
        await loadModelAndLabels();
        if (!model) return;
    }
    
    resultDiv.textContent = '⏳ Đang phân tích...';
    resultDiv.style.color = 'initial'; // Đặt lại màu mặc định

    // 1. Tiền xử lý tensor
    const tensor = tf.tidy(() => {
        // Lấy ảnh từ canvas
        const img = tf.browser.fromPixels(snapshotCanvas).toFloat();

        // 2. Thay đổi kích thước (Resize) về kích thước đầu vào của mô hình (256x256)
        const resized = tf.image.resizeBilinear(img, [MODEL_SIZE, MODEL_SIZE]);

        // 3. Chuẩn hóa giá trị pixel (MobileNetV2: [-1, 1])
        const normalized = resized.div(127.5).sub(1); 

        // 4. Mở rộng chiều batch: (H, W, C) -> (1, H, W, C)
        const batched = normalized.expandDims(0); 

        return batched;
    });

    // 5. Chạy dự đoán
    const predictions = await model.predict(tensor).data();
    
    // 6. Xử lý kết quả dự đoán
    const maxPrediction = Math.max(...predictions);
    const maxIndex = predictions.indexOf(maxPrediction);
    const predictedClass = classNames[maxIndex];
    const confidence = maxPrediction * 100;

    // 7. Hiển thị kết quả
    let resultText = '';
    
    // Tách tên bệnh/cây để hiển thị dễ đọc hơn
    const [plant, disease] = predictedClass.split('___');
    
    if (disease === 'healthy') {
        resultText = `✅ Cây **${plant}** khỏe mạnh! (Độ tin cậy: ${confidence.toFixed(2)}%)`;
        resultDiv.style.color = '#2e7d32'; // Xanh lá
    } else {
        // Định dạng tên bệnh cho dễ đọc
        const readableDisease = disease.replace(/_/g, ' ') 
            .toLowerCase()
            .replace(/(^\w|\s\w)/g, m => m.toUpperCase());
        
        resultText = `💔 Cây **${plant}** có khả năng bị bệnh **${readableDisease}**! (Độ tin cậy: ${confidence.toFixed(2)}%)`;
        resultDiv.style.color = '#d32f2f'; // Đỏ
    }
    
    resultDiv.innerHTML = resultText;
    
    // Dọn dẹp tensor
    tensor.dispose(); 
}

// === HÀM CAMERA/TẢI ẢNH ===

async function startCamera() {
    try {
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.style.display = 'block';
        
        let previewImg = document.getElementById('uploaded-image-preview');
        if (previewImg) previewImg.style.display = 'none';

        resultDiv.textContent = '✅ Camera đã sẵn sàng.';
        resultDiv.style.color = 'initial';
    } catch (err) {
        console.error("Lỗi khi truy cập camera:", err);
        resultDiv.textContent = '❌ Không thể truy cập camera. Vui lòng kiểm tra quyền.';
        resultDiv.style.color = '#d32f2f';
        video.style.display = 'none'; 
    }
}

captureButton.addEventListener('click', () => {
   
    if (video.srcObject) {
        // 1. Chụp ảnh vào Canvas
        snapshotCanvas.width = video.videoWidth;
        snapshotCanvas.height = video.videoHeight;
        const context = snapshotCanvas.getContext('2d');
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        
        // 2. Dừng camera và hiển thị ảnh xem trước
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        video.style.display = 'none';
        
        let previewImg = document.getElementById('uploaded-image-preview');
        if (!previewImg) {
            previewImg = document.createElement('img');
            previewImg.id = 'uploaded-image-preview';
            cameraSection.appendChild(previewImg);
        }
        previewImg.src = snapshotCanvas.toDataURL('image/jpeg');
        previewImg.style.display = 'block';

        // 3. Chạy phân tích
        predictImage();
        
    } else {
        resultDiv.textContent = '⚠️ Camera chưa sẵn sàng hoặc đã bị tắt.';
    }
});

fileUploadInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
    
            // Dừng camera nếu đang chạy
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
            video.style.display = 'none';
            
            // Hiển thị ảnh xem trước
            let previewImg = document.getElementById('uploaded-image-preview');
            if (!previewImg) {
                previewImg = document.createElement('img');
                previewImg.id = 'uploaded-image-preview';
                cameraSection.appendChild(previewImg);
            }
            
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            
            
            const img = new Image();
            img.onload = function() {
                // Vẽ ảnh lên Canvas để chuẩn bị phân tích
                snapshotCanvas.width = img.width;
                snapshotCanvas.height = img.height;
                const context = snapshotCanvas.getContext('2d');
                context.drawImage(img, 0, 0);
                
                // Chạy phân tích
                predictImage();
            };
            img.src = e.target.result;

            resultDiv.textContent = `⬆️ Đã tải lên "${file.name}". Đang phân tích...`;
        };
        
        reader.readAsDataURL(file);
    }
});


window.onload = loadModelAndLabels;