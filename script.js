// =========================================================================
// SCRIPT.JS - CODE HOÀN CHỈNH CHO HỆ THỐNG IPDM (EFFICIENTNET)
// =========================================================================

let model;
let disease_protocols_map = {}; // Chứa dữ liệu phác đồ chi tiết từ protocols.json
let class_indices = {}; // Chứa ánh xạ nhãn bệnh từ class_indices.json
let currentStream;

// --- 1. DOM ELEMENTS ---
const fileUpload = document.getElementById('uploadImage');
const img = document.getElementById('image');
const boxResult = document.getElementById('boxResult');
const predClassSpan = document.querySelector('.pred_class');
const confidenceSpan = document.querySelector('.confidence');
// Giả định các ID này đã có trong index.html
const mainStatus = document.getElementById('mainStatus'); 
const loadingPredictionBar = document.getElementById('loadingPredictionBar'); 

const cameraToggle = document.getElementById('cameraToggle');
const cameraContainer = document.getElementById('cameraContainer');
const videoStream = document.getElementById('videoStream');
const captureButton = document.getElementById('captureButton');
const stopButton = document.getElementById('stopButton');
const cameraStatus = document.getElementById('cameraStatus');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

const modeToggle = document.getElementById('modeToggle');
const body = document.body;

// Lấy các phần tử chi tiết từ HTML (Đã sửa ID trong index.html)
const symptomsDetail = document.getElementById('symptomsDetail'); 
const protocolSteps = document.getElementById('protocolSteps');
const resultContainer = document.getElementById('resultContainer'); 

// Khởi tạo Progress Bar
const progressBar = new ProgressBar.Circle('#progress', {
    color: '#00a896', 
    strokeWidth: 10,
    duration: 1000,
    easing: 'easeInOut',
    trailColor: '#e0e0e0', 
    trailWidth: 4,
    svgStyle: null
});


// --- 2. UTILITY FUNCTIONS ---

/**
 * Hàm làm đẹp văn bản: Tự động xử lý in đậm (từ **text** sang <b>text</b>).
 * Việc xuống dòng (<br>) được xử lý bằng cách chèn trực tiếp vào JSON.
 * @param {string} text - Chuỗi văn bản từ JSON.
 * @returns {string} - Chuỗi HTML đã được định dạng.
 */
function formatText(text) {
    if (!text) return "";
    // Xử lý in đậm: Sử dụng Regex để tìm và thay thế **text** thành <b>text</b>
    let formatted = text.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
    return formatted;
}


// --- 3. MODEL & DATA LOADING ---

async function fetchData(){
    try {
        if(mainStatus) mainStatus.textContent = 'Đang tải mô hình AI... (1/3)';
        
        // Tải mô hình EfficientNet đã chuyển đổi
        model = await tf.loadLayersModel('./tfjs_model_efficientnet/model.json');

        if(mainStatus) mainStatus.textContent = 'Đang tải nhãn và phác đồ... (2/3)';
        // Tải file nhãn (class_indices)
        const classResponse = await fetch('class_indices.json');
        class_indices = await classResponse.json();
        
        // Tải phác đồ quản lý (Giả định file này có tên là protocols.json)
        const protocolsResponse = await fetch('protocols.json'); 
        const protocols = await protocolsResponse.json();

        // Xây dựng map dễ truy cập
        protocols.Phac_do_Quan_Ly_Tong_hop_Chi_tiet.forEach(item => {
            disease_protocols_map[item.Tên_Bệnh] = item;
        });

        if(mainStatus) mainStatus.textContent = '✅ Hệ thống sẵn sàng. Tải ảnh hoặc bật Camera.';
        cameraToggle.disabled = false;
        fileUpload.disabled = false;
        
    } catch (error) {
        console.error("❌ Lỗi khi tải tài nguyên:", error);
        if(mainStatus) mainStatus.textContent = '❌ Lỗi tải tài nguyên. Vui lòng kiểm tra file model.json, class_indices.json và protocols.json.';
    }
}


// --- 4. HÀM DỰ ĐOÁN (CỐT LÕI - ĐÃ FIX CHUẨN HÓA EFFICIENTNET) ---

async function predict(imageElement) {
    if (!model) {
        alert("Model chưa được load!");
        return;
    }

    loadingPredictionBar.style.display = 'flex';
    boxResult.style.display = 'none';

    try {
        const tensor = tf.tidy(() => {
            // 1. Chuyển ảnh DOM thành Tensor
            let img = tf.browser.fromPixels(imageElement)
                .resizeNearestNeighbor([224, 224]) // Resize chuẩn 224x224
                .toFloat();

            // 2. CHUẨN HÓA (QUAN TRỌNG: Chia 255 để khớp với quá trình train EfficientNet)
            const normalized = img.div(tf.scalar(255.0)); 

            // 3. Thêm chiều batch [1, 224, 224, 3]
            return normalized.expandDims();
        });

        // Dự đoán
        const predictions = await model.predict(tensor).data();
        tensor.dispose(); 

        // Tìm kết quả lớn nhất
        let maxPrediction = -1;
        let maxIndex = -1;
        for (let i = 0; i < predictions.length; i++) {
            if (predictions[i] > maxPrediction) {
                maxPrediction = predictions[i];
                maxIndex = i;
            }
        }

        displayResult(maxIndex, maxPrediction);

    } catch (error) {
        console.error("Lỗi khi dự đoán:", error);
        alert("Có lỗi xảy ra khi phân tích ảnh.");
    } finally {
        loadingPredictionBar.style.display = 'none';
    }
}

// --- 5. HÀM HIỂN THỊ KẾT QUẢ ---

function displayResult(classIndex, confidence) {
    // Tìm tên bệnh
    const predictedClassName = Object.keys(class_indices).find(key => class_indices[key] === classIndex);
    predClassSpan.textContent = predictedClassName || "Không xác định";
    confidenceSpan.textContent = (confidence * 100).toFixed(2);
    
    progressBar.animate(confidence);

    const protocol = disease_protocols_map[predictedClassName];
    boxResult.style.display = 'block';

    if (protocol) {
        // CẬP NHẬT DẤU HIỆU CHUYÊN SÂU
        if (protocol.Dấu_hiệu_Chẩn_đoán_Chuyên_sâu) {
            // SỬ DỤNG .innerHTML và formatText để xử lý **in đậm** và <br><br>
            symptomsDetail.innerHTML = formatText(protocol.Dấu_hiệu_Chẩn_đoán_Chuyên_sâu);
        } else {
            symptomsDetail.innerHTML = "Không có mô tả dấu hiệu chi tiết.";
        }
        
        // CẬP NHẬT CÁC BƯỚC PHÁC ĐỒ (Ví dụ cho mục I_Tác_nhân_Chu_kỳ_và_Điều_kiện)
        if(protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện) {
            let protocolHtml = `
                <p><strong>Tác nhân:</strong> ${formatText(protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Tác_nhân_Sinh_học)}</p>
                <p><strong>Cơ chế lây lan:</strong> ${formatText(protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Cơ_chế_Lây_lan)}</p>
                <p><strong>Thời điểm tối ưu:</strong> ${formatText(protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Nhiệt_độ_Thời_điểm_tối_ưu)}</p>
            `;
            // Bạn cần mở rộng phần này để hiển thị các mục II, III, IV
            protocolSteps.innerHTML = protocolHtml;
        }


    } else {
        symptomsDetail.innerHTML = "Không tìm thấy phác đồ quản lý cho bệnh này.";
        protocolSteps.innerHTML = "Không tìm thấy phác đồ quản lý cho bệnh này.";
    }
}


// --- 6. HÀM XỬ LÝ CAMERA/UI (GIỮ NGUYÊN LOGIC CŨ) ---

function startCamera() {
    const constraints = {
        video: {
            facingMode: "environment", // Ưu tiên camera sau trên di động
            width: 224,
            height: 224
        }
    };

    navigator.mediaDevices.getUserMedia(constraints)
        .then(function(stream) {
            currentStream = stream;
            videoStream.srcObject = stream;
            videoStream.play();
            
            videoStream.style.display = 'block';
            captureButton.style.display = 'inline-block';
            stopButton.style.display = 'inline-block';
            img.style.display = 'none'; 
            
            cameraStatus.textContent = 'Camera đang hoạt động. Hãy chụp lá cây.';
            
            fileUpload.disabled = true; // Vô hiệu hóa upload khi camera bật
        })
        .catch(function(err) {
            console.error("Lỗi khi truy cập camera: " + err.name + ": " + err.message);
            cameraStatus.textContent = '❌ Lỗi truy cập Camera. Kiểm tra quyền truy cập.';
        });
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    videoStream.style.display = 'none';
    captureButton.style.display = 'none';
    stopButton.style.display = 'none';
    img.style.display = 'block'; 
    cameraStatus.textContent = 'Camera đã tắt.';
    fileUpload.disabled = false; // Bật lại upload
}

// --- 7. EVENT LISTENERS ---

fileUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        stopCamera(); 
        const reader = new FileReader();
        reader.onload = function(e) {
            img.src = e.target.result;
            img.onload = () => {
                predict(img);
            };
        };
        reader.readAsDataURL(file);
    }
});


cameraToggle.addEventListener('click', function() {
    if (!currentStream) {
        startCamera();
    } else {
        stopCamera();
    }
});

captureButton.addEventListener('click', () => {
    // 1. Chụp ảnh từ video stream
    canvas.width = 224;
    canvas.height = 224;
    context.drawImage(videoStream, 0, 0, canvas.width, canvas.height); 
    
    // 2. Cập nhật ảnh vào thẻ <img>
    const dataUrl = canvas.toDataURL('image/png');
    img.src = dataUrl;
    
    // 3. Ẩn camera UI
    videoStream.style.display = 'none';
    captureButton.style.display = 'none';
    stopButton.style.display = 'none';
    img.style.display = 'block'; 
    stopCamera(); 
    
    cameraStatus.textContent = 'Ảnh đã được chụp. Đang phân tích...';
    
    // 4. Dự đoán
    predict(img);
});


stopButton.addEventListener('click', stopCamera);


// Xử lý Dark/Light mode
modeToggle.addEventListener('click', () => {
    const isLightMode = body.classList.contains('light-mode');
    if (isLightMode) {
        body.classList.replace('light-mode', 'dark-mode');
        modeToggle.innerHTML = '<i class="material-icons">wb_sunny</i> Chế độ Sáng';
    } else {
        body.classList.replace('dark-mode', 'light-mode');
        modeToggle.innerHTML = '<i class="material-icons">brightness_4</i> Chế độ Tối';
    }
});


// Khởi chạy khi tải trang
window.onload = fetchData;
