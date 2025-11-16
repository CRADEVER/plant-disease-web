
const video = document.getElementById('camera');
const captureButton = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const fileUploadInput = document.getElementById('file-upload');
const cameraSection = document.getElementById('camera-section'); 
const previewImg = document.getElementById('uploaded-image-preview');

const MODEL_URL = 'plant_model_js/model.json'; 
const IMAGE_SIZE = 256; 

let model;

let classNames = window.CLASS_NAMES || []; 

async function loadModel() {
    resultDiv.textContent = '⏳ Đang tải mô hình AI...';
    try {

        model = await tf.loadLayersModel(MODEL_URL);
        
        if (classNames.length === 0) {
             resultDiv.textContent = '❌ Lỗi: Không tìm thấy nhãn CLASS_NAMES. Vui lòng kiểm tra file labels.js.';
             return;
        }

        resultDiv.textContent = `✅ Mô hình đã sẵn sàng. (${classNames.length} loại bệnh)`;
    } catch (err) {
      
        console.error("Lỗi khi tải mô hình:", err);
        resultDiv.textContent = '❌ Lỗi khi tải mô hình AI. Vui lòng kiểm tra đường dẫn hoặc file model.json.';
    }
}

async function runModelPrediction() {
    if (!model) {
        resultDiv.textContent = '⚠️ Mô hình chưa được tải. Đang tải lại...';
        await loadModel();
        if (!model) return;
    }
    
    resultDiv.textContent = '🧠 Đang phân tích hình ảnh...';
    
  
    let tensor = tf.browser.fromPixels(snapshotCanvas)
        .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE]) 
        .toFloat();

  
    tensor = tf.sub(tf.div(tensor, 127.5), 1);
    
    const expandedTensor = tensor.expandDims(0);
    
    const prediction = await model.predict(expandedTensor).data();
    
    const predictedClassIndex = prediction.indexOf(Math.max(...prediction));
    const predictedClass = classNames[predictedClassIndex] || 'Không xác định';
    const confidence = prediction[predictedClassIndex] * 100;


    let resultText = `Kết quả: **${predictedClass}**`;
    
    if (predictedClass.toLowerCase().includes('healthy')) {
        resultText = `**Phân loại:** Cây khỏe mạnh 🎉 (**${predictedClass}**)`;
    } else if (predictedClass !== 'Không xác định') {
        resultText = `**Phân loại:** Cây bị bệnh! 🚨 (**${predictedClass}**)`;
    }
    
    resultDiv.innerHTML = `${resultText}<br>**Độ tin cậy:** ${confidence.toFixed(2)}%`;

    // Giải phóng bộ nhớ Tensor
    tf.dispose([tensor, expandedTensor]);
}




async function startCamera() {
    
    await loadModel();

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        
    } catch (err) {
        console.error("Lỗi khi truy cập camera:", err);
        resultDiv.textContent = '❌ Không thể truy cập camera. Vui lòng kiểm tra quyền.';
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
        video.style.display = 'none';
        previewImg.src = imageDataURL;
        previewImg.style.display = 'block';

        resultDiv.textContent = '📸 Ảnh đã chụp. Đang phân tích...';
        runModelPrediction();

    } else {
        resultDiv.textContent = '⚠️ Camera chưa sẵn sàng hoặc đã bị tắt.';
    }
});

fileUploadInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            video.style.display = 'none';
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
     
            const img = new Image();
            img.onload = function() {
                snapshotCanvas.width = img.width;
                snapshotCanvas.height = img.height;
                const context = snapshotCanvas.getContext('2d');
                context.drawImage(img, 0, 0);
                

                runModelPrediction();
            };
            img.src = e.target.result;

            resultDiv.textContent = `⬆️ Đã tải lên "${file.name}". Đang phân tích...`;
        };
        
        reader.readAsDataURL(file);
    }
});



window.onload = startCamera;
