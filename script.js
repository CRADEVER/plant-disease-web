// script.js (Phiên bản Hoàn Chỉnh - Tích hợp TensorFlow.js thực tế)

let model;
let disease_protocols_map = {}; // Lưu trữ map các phác đồ để tra cứu nhanh bằng Mã_ID
let class_indices = {}; // Ánh xạ Mã ID -> Tên Bệnh
let currentStream;

// === KHAI BÁO CÁC PHẦN TỬ DOM ===
const fileUpload = document.getElementById('uploadImage');
const img = document.getElementById('image');
const boxResult = document.getElementById('boxResult');
const predClassSpan = document.querySelector('.pred_class');
const confidenceSpan = document.querySelector('.confidence');
const resultContainer = document.getElementById('resultContainer'); 
const mainStatus = document.getElementById('mainStatus'); 
const loadingPredictionBar = document.getElementById('loadingPredictionBar'); 

// Camera/Video elements
const cameraToggle = document.getElementById('cameraToggle');
const cameraContainer = document.getElementById('cameraContainer');
const videoStream = document.getElementById('videoStream');
const captureButton = document.getElementById('captureButton');
const stopButton = document.getElementById('stopButton');
const cameraStatus = document.getElementById('cameraStatus');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

// Mode toggle
const modeToggle = document.getElementById('modeToggle');
const body = document.body;

// Cấu hình ProgressBar (Yêu cầu thư viện progressbar.js)
const progressBar = new ProgressBar.Circle('#progress', {
    color: 'limegreen',
    strokeWidth: 10,
    duration: 1000,
    easing: 'easeInOut',
    trailColor: '#e0e0e0',
    trailWidth: 4,
    svgStyle: null
});


// === LOGIC XỬ LÝ DỮ LIỆU ===

async function fetchData(){
    try {
        let response = await fetch('./class_indices.json');
        let data = await response.json();
        
        let protocolMap = {};
        let indicesMap = {};
        
        // Chuyển đổi mảng JSON thành map tra cứu (Mã_ID -> Phác đồ chi tiết)
        data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet.forEach(item => {
            protocolMap[item.Mã_ID] = item;
            indicesMap[item.Mã_ID] = item.Tên_Bệnh; 
        });

        disease_protocols_map = protocolMap;
        class_indices = indicesMap; 

    } catch (error) {
        console.error("Lỗi khi tải class_indices.json:", error);
        mainStatus.className = 'status error';
        mainStatus.innerHTML = '<i class="material-icons">error_outline</i> Lỗi: Không thể tải phác đồ quản lý bệnh.';
        return null;
    }
}


async function initialize() {
    mainStatus.className = 'status loading';
    mainStatus.innerHTML = '<i class="material-icons loading-icon">cached</i> Đang tải mô hình và dữ liệu quản lý...';
    
    // 1. Tải dữ liệu JSON
    await fetchData();

    // 2. Tải mô hình THỰC TẾ
    try {
        // ĐƯỜNG DẪN ĐƯỢC CHỈNH SỬA THEO YÊU CẦU CỦA BẠN
        const modelUrl = './tensorflowjs-model/model.json'; 
        model = await tf.loadLayersModel(modelUrl); 
        console.log("Mô hình đã tải thành công:", model);

        mainStatus.className = 'status success';
        mainStatus.innerHTML = '<i class="material-icons">check_circle_outline</i> Hệ thống đã sẵn sàng. Hãy chọn ảnh hoặc dùng Camera.';
        
    } catch (error) {
        console.error("Lỗi khi tải mô hình TensorFlow.js:", error);
        mainStatus.className = 'status error';
        mainStatus.innerHTML = '<i class="material-icons">error_outline</i> Lỗi: Không thể tải mô hình dự đoán. Đảm bảo thư mục <b>tensorflowjs-model</b> chứa <b>model.json</b> và các file <b>.bin</b>.';
    }
}


// HÀM HIỂN THỊ CHI TIẾT PHÁC ĐỒ
function displayDiseaseDetails(protocol) {
    resultContainer.style.display = 'block';
    
    // Tạo cấu trúc HTML chi tiết, sử dụng thẻ <details>
    let html = `
        <div class="protocol-header">
            <h3>${protocol.Tên_Bệnh}</h3>
            <p class="classification">Phân loại: <b>${protocol.Phân_loại}</b></p>
        </div>
        <hr>
        
        <div class="protocol-sections">
            <details class="protocol-detail-section" open>
                <summary>
                    <i class="material-icons">science</i>
                    I. Tác nhân, Chu kỳ và Điều kiện (Cơ sở)
                </summary>
                <div class="detail-content">
                    <p><b>Tác nhân:</b> ${protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Tác_nhân_Sinh_học}</p>
                    <p><b>Cơ chế lây lan:</b> ${protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Cơ_chế_Lây_lan}</p>
                    <p><b>Nhiệt độ tối ưu:</b> ${protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Nhiệt_độ_Thời_điểm_tối_ưu}</p>
                    <p><b>Dấu hiệu:</b> ${protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Dấu_hiệu_Chẩn_đoán_Chuyên_sâu}</p>
                </div>
            </details>

            <details class="protocol-detail-section">
                <summary>
                    <i class="material-icons">agriculture</i>
                    II. Chiến lược Văn hóa và Cơ học (Phòng ngừa)
                </summary>
                <div class="detail-content">
                    <ul>
                        <li><b>Quản lý Giống & Cây trồng:</b> ${protocol.II_Chiến_lược_Văn_hóa_Cơ_học.Quản_lý_Giống_Cây_trồng}</li>
                        <li><b>Quản lý Dinh dưỡng & Nước:</b> ${protocol.II_Chiến_lược_Văn_hóa_Cơ_học.Quản_lý_Dinh_dưỡng_và_Nước}</li>
                        <li><b>Vệ sinh đồng ruộng:</b> ${protocol.II_Chiến_lược_Văn_hóa_Cơ_học.Vệ_sinh_Đồng_ruộng_Tiểu_khí_hậu}</li>
                    </ul>
                </div>
            </details>

            <details class="protocol-detail-section">
                <summary>
                    <i class="material-icons">local_florist</i>
                    III. Chiến lược Kiểm soát Hóa học (Thuốc)
                </summary>
                <div class="detail-content">
                    <p><b>Nguyên tắc FRAC/IRAC:</b> ${protocol.III_Chiến_lược_Kiểm_soát_Hóa_học.Nguyên_tắc_FRAC_IRAC}</p>
                    
                    <h4>Phác đồ theo Giai đoạn:</h4>
                    ${protocol.III_Chiến_lược_Kiểm_soát_Hóa_học.Phác_đồ_Giai_đoạn_Cây.map(step => `
                        <div class="stage-step">
                            <p><b>Giai đoạn:</b> ${step.Giai_đoạn}</p>
                            <p><b>Hoạt chất đề xuất:</b> <span>${step.Hoạt_chất_Đề_xuất}</span> (Nhóm: ${step.Nhóm_FRAC_IRAC})</p>
                            <p><b>Lưu ý:</b> ${step.Lưu_ý_Ứng_dụng}</p>
                        </div>
                    `).join('')}
                </div>
            </details>
            
            <details class="protocol-detail-section">
                <summary>
                    <i class="material-icons">hive</i>
                    IV. Giải pháp Sinh học và Khác
                </summary>
                <div class="detail-content">
                    <p><b>Giải pháp Sinh học:</b> ${protocol.IV_Giải_pháp_Sinh_học_và_Khác.Giải_pháp_Sinh_học}</p>
                    <p><b>Quản lý Kháng thuốc (IRM):</b> ${protocol.IV_Giải_pháp_Sinh_học_và_Khác.Quản_lý_Kháng_thuốc_IRM}</p>
                </div>
            </details>
        </div>
    `;

    resultContainer.innerHTML = html;
}


// HÀM DỰ ĐOÁN THỰC TẾ (Sử dụng TensorFlow.js)
async function predict(imageElement) {
    if (!model) {
        mainStatus.className = 'status error';
        mainStatus.innerHTML = '<i class="material-icons">error_outline</i> Mô hình chưa được tải. Vui lòng kiểm tra console.';
        return;
    }
    
    // UI Loading
    resultContainer.style.display = 'none';
    boxResult.style.display = 'flex'; 
    loadingPredictionBar.style.display = 'flex'; 
    progressBar.set(0);
    confidenceSpan.textContent = 0;
    predClassSpan.textContent = 'Đang phân tích...';
    
    let predicted_index, confidence_score;
    
    try {
        // 1. Tiền xử lý ảnh (Resize 224x224, Chuẩn hóa, Thêm chiều batch)
        const tensor = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([224, 224]) // Kích thước input thường thấy
            .toFloat()
            .div(tf.scalar(255.0)) // Chuẩn hóa về [0, 1]
            .expandDims(); // Thêm chiều batch

        // 2. Chạy dự đoán
        const predictions = model.predict(tensor);
        const predictionArray = await predictions.data();
        
        // 3. Xử lý kết quả
        const highestPrediction = Math.max(...predictionArray);
        predicted_index = predictionArray.indexOf(highestPrediction).toString(); 
        confidence_score = Math.floor(highestPrediction * 100);

        // Giải phóng tensor để tránh rò rỉ bộ nhớ
        tensor.dispose(); 
        predictions.dispose();

    } catch (e) {
        console.error("Lỗi khi chạy dự đoán:", e);
        loadingPredictionBar.style.display = 'none';
        predClassSpan.textContent = 'Lỗi Phân Tích!';
        confidenceSpan.textContent = 0;
        resultContainer.style.display = 'block';
        resultContainer.innerHTML = `<div class="protocol-header error">
            <i class="material-icons">warning</i> 
            Lỗi trong quá trình xử lý ảnh và dự đoán. Vui lòng kiểm tra console.
        </div>`;
        return;
    }
    
    // Ẩn loading và hiển thị kết quả
    loadingPredictionBar.style.display = 'none';

    let normalizedConfidence = confidence_score / 100;
    progressBar.animate(normalizedConfidence, () => {
        confidenceSpan.textContent = confidence_score;
    });

    const diseaseName = class_indices[predicted_index] || "Không xác định (Mã: " + predicted_index + ")";
    predClassSpan.textContent = diseaseName;
    
    const protocol = disease_protocols_map[predicted_index];

    if (protocol) {
        displayDiseaseDetails(protocol);
    } else {
        resultContainer.innerHTML = `<div class="protocol-header error">
            <i class="material-icons">warning</i> 
            Không tìm thấy phác đồ quản lý chi tiết cho bệnh <b>${diseaseName}</b>.
        </div>`;
        resultContainer.style.display = 'block';
    }
}


// === LOGIC CAMERA VÀ XỬ LÝ SỰ KIỆN (Giữ nguyên) ===

async function startCamera() {
    boxResult.style.display = 'none';
    resultContainer.style.display = 'none';
    img.style.display = 'none';
    document.querySelector('.image-placeholder').style.display = 'none';
    
    cameraContainer.style.display = 'block';

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            currentStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            videoStream.srcObject = currentStream;
            videoStream.play();
            cameraStatus.textContent = 'Camera đã sẵn sàng. Hãy chụp ảnh.';
            captureButton.disabled = false;
            videoStream.style.display = 'block';
            captureButton.style.display = 'flex';
            stopButton.style.display = 'flex';
        } catch (error) {
            cameraStatus.textContent = `Lỗi truy cập camera: ${error.name}. Vui lòng đảm bảo camera được phép sử dụng.`;
            captureButton.disabled = true;
            videoStream.style.display = 'none';
            captureButton.style.display = 'none';
            stopButton.style.display = 'none';
            cameraContainer.style.display = 'block';
        }
    } else {
        cameraStatus.textContent = 'Trình duyệt không hỗ trợ Media Devices API.';
        cameraContainer.style.display = 'block';
    }
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    videoStream.srcObject = null;
    captureButton.disabled = true;
    cameraStatus.textContent = 'Camera đã dừng.';
    cameraContainer.style.display = 'none';
    captureButton.style.display = 'none';
    stopButton.style.display = 'none';
    document.querySelector('.image-placeholder').style.display = 'block';
}


// Đăng ký sự kiện Tải ảnh
fileUpload.addEventListener('change', function () {
    const file = this.files[0];
    if (file) {
        stopCamera();
        const reader = new FileReader();
        reader.onload = function (e) {
            img.src = e.target.result;
            img.style.display = 'block'; 
            document.querySelector('.image-placeholder').style.display = 'none';
            img.onload = () => {
                predict(img);
            };
        };
        reader.readAsDataURL(file);
    }
});


// Đăng ký sự kiện Camera
cameraToggle.addEventListener('click', function() {
    if (!currentStream) {
        startCamera();
    } else {
        stopCamera();
    }
});

// Chụp ảnh từ Camera
captureButton.addEventListener('click', () => {
    resultContainer.style.display = 'none'; 
    
    canvas.width = 224;
    canvas.height = 224;
    context.drawImage(videoStream, 0, 0, canvas.width, canvas.height); 
    
    const dataUrl = canvas.toDataURL('image/png');
    img.src = dataUrl;
    img.style.display = 'block'; 
    
    videoStream.style.display = 'none';
    captureButton.style.display = 'none';
    stopButton.style.display = 'none';
    cameraContainer.style.display = 'none';
    
    cameraStatus.textContent = 'Ảnh đã được chụp. Đang phân tích...';
    
    predict(img);
});


// Logic Dark/Light Mode
modeToggle.addEventListener('click', () => {
    if (body.classList.contains('light-mode')) {
        body.classList.replace('light-mode', 'dark-mode');
        modeToggle.innerHTML = '<i class="material-icons">wb_sunny</i> Chế độ Sáng';
        progressBar.options.trailColor = '#333333';
        progressBar.set(progressBar.value()); 
    } else {
        body.classList.replace('dark-mode', 'light-mode');
        modeToggle.innerHTML = '<i class="material-icons">brightness_4</i> Chế độ Tối';
        progressBar.options.trailColor = '#e0e0e0';
        progressBar.set(progressBar.value()); 
    }
});


// Khởi chạy hệ thống khi DOM đã tải
document.addEventListener('DOMContentLoaded', initialize);