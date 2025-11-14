const video = document.getElementById('camera');
const captureButton = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const fileUploadInput = document.getElementById('file-upload');
const cameraSection = document.getElementById('camera-section');

// Kích thước đầu vào của mô hình (đã xác định là 256 trong file .ipynb)
const MODEL_SIZE = 256; 
let model;
let classNames = []; 

// === HÀM TẢI MÔ HÌNH VÀ LABELS ===
async function loadModelAndLabels() {
    try {
        // Tải labels từ file labels.js (đã tạo ở bước cuối trong Colab)
        await import('./plant_model_js/labels.js').then(module => {
            classNames = module.CLASS_NAMES;
            console.log(`Đã tải ${classNames.length} classes.`);
        });
        
        // Tải mô hình TF.js
        model = await tf.loadLayersModel('./plant_model_js/model.json');
        console.log("✅ Model TF.js đã tải thành công.");
        
        // Bắt đầu camera sau khi model đã sẵn sàng
        startCamera();
    } catch (error) {
        console.error("Lỗi khi tải model hoặc labels:", error);
        resultDiv.textContent = '❌ Lỗi: Không thể tải mô hình phân tích.';
        // Vẫn gọi startCamera để cho phép người dùng chụp ảnh/tải ảnh nếu lỗi không nghiêm trọng
        startCamera(); 
    }
}

// === HÀM PHÂN TÍCH ẢNH ===
async function predictImage() {
    if (!model || classNames.length === 0) {
        resultDiv.textContent = '⚠️ Mô hình chưa sẵn sàng. Đang tải lại...';
        await loadModelAndLabels();
        if (!model) return;
    }
    
    resultDiv.textContent = '⏳ Đang phân tích...';

    // 1. Tiền xử lý tensor
    const tensor = tf.tidy(() => {
        // Lấy ảnh từ canvas (đã có ảnh chụp/tải lên)
        const img = tf.browser.fromPixels(snapshotCanvas).toFloat();

        // 2. Thay đổi kích thước (Resize) về kích thước đầu vào của mô hình
        const resized = tf.image.resizeBilinear(img, [MODEL_SIZE, MODEL_SIZE]);

        // 3. Chuẩn hóa giá trị pixel (-1 đến 1 theo MobileNetV2 preprocess_input)
        // Đây là cách chuẩn hóa MobileNetV2: (x/127.5) - 1
        const normalized = resized.div(127.5).sub(1); 

        // 4. Mở rộng chiều batch: (height, width, channels) -> (1, height, width, channels)
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
    if (predictedClass.includes('healthy')) {
        resultText = `✅ Cây **khỏe mạnh**! (Độ tin cậy: ${confidence.toFixed(2)}%)`;
        resultDiv.style.color = '#2e7d32'; // Xanh lá
    } else {
        // Thay thế dấu gạch dưới và viết hoa chữ cái đầu cho tên bệnh dễ đọc hơn
        const readableClass = predictedClass
            .split('___')[1] // Lấy tên bệnh
            .replace(/_/g, ' ') // Thay gạch dưới bằng khoảng trắng
            .toLowerCase()
            .replace(/(^\w|\s\w)/g, m => m.toUpperCase()); // Viết hoa chữ cái đầu
        
        resultText = `💔 Cây có khả năng bị bệnh **${readableClass}**! (Độ tin cậy: ${confidence.toFixed(2)}%)`;
        resultDiv.style.color = '#d32f2f'; // Đỏ
    }
    
    resultDiv.innerHTML = resultText;
    
    // Dọn dẹp tensor
    tensor.dispose(); 
}

// === HÀM CAMERA/TẢI ẢNH (Không đổi, chỉ thêm gọi hàm phân tích) ===

async function startCamera() {
    try {
        // Tắt video nếu đang chạy stream cũ
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.style.display = 'block';
        
        let previewImg = document.getElementById('uploaded-image-preview');
        if (previewImg) previewImg.style.display = 'none';

        resultDiv.textContent = '✅ Camera đã sẵn sàng.';
        resultDiv.style.color = 'initial'; // Reset màu
    } catch (err) {
        console.error("Lỗi khi truy cập camera:", err);
        resultDiv.textContent = '❌ Không thể truy cập camera. Vui lòng kiểm tra quyền.';
        resultDiv.style.color = '#d32f2f';
        video.style.display = 'none'; 
    }
}

captureButton.addEventListener('click', () => {
   
    if (video.srcObject) {

        snapshotCanvas.width = video.videoWidth;
        snapshotCanvas.height = video.videoHeight;
     
        const context = snapshotCanvas.getContext('2d');
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        
        
        const imageDataURL = snapshotCanvas.toDataURL('image/jpeg');
        
        // Tắt stream camera (tùy chọn)
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
        previewImg.src = imageDataURL;
        previewImg.style.display = 'block';

        resultDiv.textContent = '📸 Ảnh đã chụp. Sẵn sàng phân tích.';
        
        // GỌI HÀM PHÂN TÍCH SAU KHI CHỤP
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
    
            // Tắt stream camera (tùy chọn)
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
            
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            
            
            const img = new Image();
            img.onload = function() {
                // Đảm bảo canvas có kích thước chính xác của ảnh tải lên
                snapshotCanvas.width = img.width;
                snapshotCanvas.height = img.height;
                const context = snapshotCanvas.getContext('2d');
                context.drawImage(img, 0, 0);
                
                // GỌI HÀM PHÂN TÍCH SAU KHI TẢI VÀ VẼ LÊN CANVAS
                predictImage();
            };
            img.src = e.target.result;

            resultDiv.textContent = `⬆️ Đã tải lên "${file.name}". Sẵn sàng phân tích.`;
        };
        
        reader.readAsDataURL(file);
    }
});


window.onload = loadModelAndLabels;