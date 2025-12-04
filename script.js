let model;
let disease_protocols_map = {}; 
let class_indices = {}; 
let currentStream;

// --- DOM ELEMENTS ---
const fileUpload = document.getElementById('uploadImage');
const img = document.getElementById('image');
const boxResult = document.getElementById('boxResult');
const predClassSpan = document.querySelector('.pred_class');
const confidenceSpan = document.querySelector('.confidence');
const resultContainer = document.getElementById('resultContainer'); 
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

// Progress bar setup
const progressBar = new ProgressBar.Circle('#progress', {
    color: '#00a896', 
    strokeWidth: 8,
    duration: 1400,
    easing: 'easeInOut',
    trailColor: '#e0e0e0', 
    trailWidth: 2,
    svgStyle: null
});

// --- LOAD DATA & MODEL ---

async function fetchData(){
    try {
        let response = await fetch('./class_indices.json');
        
        if (!response.ok) {
            throw new Error(`HTTP Error! Status: ${response.status}. Hãy đảm bảo file class_indices.json nằm cùng thư mục.`);
        }
        
        let data = await response.json();
        
        let protocolMap = {};
        let indicesMap = {};
        
        // Truy cập vào mảng chính trong JSON
        const protocolsArray = data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet;

        if (Array.isArray(protocolsArray)) {
            protocolsArray.forEach(item => {
                // Mapping dựa trên Mã_ID (ví dụ: "0", "1", "2"...)
                protocolMap[item.Mã_ID] = item;
                indicesMap[item.Mã_ID] = item.Tên_Bệnh; 
            });
        } else {
             throw new Error("Cấu trúc JSON không hợp lệ: Không tìm thấy mảng 'Phac_do_Quan_Ly_Tong_Hop_Chi_tiet'.");
        }

        disease_protocols_map = protocolMap;
        class_indices = indicesMap; 
        
        console.log("DEBUG: Dữ liệu bệnh cây đã tải thành công.", { Total: protocolsArray.length });

    } catch (error) {
        console.error("Lỗi khi tải class_indices.json:", error);
        mainStatus.className = 'status error';
        mainStatus.innerHTML = `<i class="material-icons">error_outline</i> Lỗi tải dữ liệu: ${error.message}`;
    }
}

async function initialize() {
    mainStatus.className = 'status loading';
    mainStatus.innerHTML = '<i class="material-icons loading-icon">cached</i> Đang tải mô hình AI và cơ sở dữ liệu...';
    
    await fetchData();
  
    try {
        // Lưu ý: Đảm bảo đường dẫn này chính xác với cấu trúc thư mục của bạn
        const modelUrl = './tensorflowjs-model/model.json'; 
        model = await tf.loadLayersModel(modelUrl); 

        mainStatus.className = 'status success';
        mainStatus.innerHTML = '<i class="material-icons">check_circle_outline</i> Hệ thống đã sẵn sàng. Hãy chọn ảnh để phân tích.';
        console.log("DEBUG: Model loaded successfully.");
        
    } catch (error) {
        console.error("Lỗi khi tải mô hình:", error);
        mainStatus.className = 'status error';
        mainStatus.innerHTML = '<i class="material-icons">error_outline</i> Lỗi: Không thể tải mô hình (model.json). Kiểm tra thư mục tensorflowjs-model.';
    }
}

// --- DISPLAY LOGIC ---

function displayDiseaseDetails(protocol) {
    resultContainer.style.display = 'block';
    
    // 1. Phác đồ hóa học
    const phacDoGiaiDoan = protocol.III_Chiến_lược_Kiểm_soát_Hóa_học?.Phác_đồ_Giai_đoạn_Cây || [];
    let chemicalStepsHtml = '';
    
    if (phacDoGiaiDoan.length > 0) {
        chemicalStepsHtml = phacDoGiaiDoan.map(step => `
            <div class="stage-step">
                <p><b>Giai đoạn:</b> ${step.Giai_đoạn || 'N/A'}</p>
                <p><b>Hoạt chất:</b> <span>${step.Hoạt_chất_Đề_xuất || 'N/A'}</span> (Nhóm: ${step.Nhóm_FRAC_IRAC || 'N/A'})</p>
                <p><b>Lưu ý:</b> ${step.Lưu_ý_Ứng_dụng || 'N/A'}</p>
            </div>
        `).join('');
    } else {
        chemicalStepsHtml = '<p><i>Không có phác đồ hóa học cụ thể hoặc cây khỏe mạnh.</i></p>';
    }

    // 2. Canh tác chuyên sâu
    const cult = protocol.II_Biện_pháp_Canh_tác_Chuyên_sâu || {};
    let cultHtml = `
        <ul>
            <li><b>Đất & Tàn dư:</b> ${cult.Quản_lý_Tàn_dư_Đất || '...'}</li>
            <li><b>Nước tưới:</b> ${cult.Quản_lý_Nước_Tưới || '...'}</li>
            <li><b>Mật độ & Cắt tỉa:</b> ${cult.Mật_độ_Thông_thoáng || '...'}</li>
            <li><b>Dinh dưỡng:</b> ${cult.Quản_lý_Dinh_dưỡng_Vi_lượng || '...'}</li>
        </ul>
    `;

    // 3. Sinh học & Thay thế
    const bio = protocol.IV_Giải_pháp_Sinh_học_và_Thay_thế || {};
    let bioHtml = `
        <p><b>Đối kháng VSV:</b> ${bio.Chất_Đối_kháng_VSV || '...'}</p>
        <p><b>Kích kháng (SAR):</b> ${bio.Cảm_ứng_Kháng_Bệnh_SAR || '...'}</p>
        <p><b>Vector truyền bệnh:</b> ${bio.Kiểm_soát_Côn_trùng_Vector || '...'}</p>
    `;

    // 4. Nguồn thông tin (Xử lý định dạng hỗn hợp trong JSON)
    let sourcesHtml = '<ul class="source-list">';
    const sources = protocol.V_Nguồn_Thông_Tin;
    if (sources) {
        // Duyệt qua các key (1, 2, 3 hoặc Nguon_1, Nguon_2...)
        Object.keys(sources).forEach(key => {
            const item = sources[key];
            let url = '#';
            let name = 'Nguồn tham khảo';

            if (typeof item === 'string') {
                url = item;
                name = item; // Nếu chỉ là string URL
            } else if (typeof item === 'object') {
                url = item.URL || '#';
                name = item.Tên_Nguồn || item.URL || `Nguồn ${key}`;
            }

            if (url !== '#') {
                sourcesHtml += `<li><a href="${url}" target="_blank">${name}</a></li>`;
            }
        });
    }
    sourcesHtml += '</ul>';
    if (!sources) sourcesHtml = '<p><i>Đang cập nhật nguồn tham khảo.</i></p>';

    // --- HTML Template Construction ---
    let html = `
        <div class="protocol-header">
            <h3>${protocol.Tên_Bệnh || 'Chưa xác định'}</h3>
            <p class="classification">Phân loại: <b>${protocol.Phân_loại || '...'}</b></p>
        </div>
        
        <div class="protocol-sections">
            <details class="protocol-detail-section" open>
                <summary><i class="material-icons">science</i> I. Tác nhân & Điều kiện phát sinh</summary>
                <div class="detail-content">
                    <p><b>Tác nhân:</b> ${protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện?.Tác_nhân_Sinh_học || '...'}</p>
                    <p><b>Cơ chế lây lan:</b> ${protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện?.Cơ_chế_Lây_lan || '...'}</p>
                    <p><b>Điều kiện tối ưu:</b> ${protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện?.Nhiệt_độ_Thời_điểm_tối_ưu || '...'}</p>
                    <p><b>Dấu hiệu chẩn đoán:</b> ${protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện?.Dấu_hiệu_Chẩn_đoán_Chuyên_sâu || '...'}</p>
                </div>
            </details>

            <details class="protocol-detail-section">
                <summary><i class="material-icons">agriculture</i> II. Biện pháp Canh tác (Phòng ngừa)</summary>
                <div class="detail-content">${cultHtml}</div>
            </details>

            <details class="protocol-detail-section">
                <summary><i class="material-icons">local_florist</i> III. Kiểm soát Hóa học (Trị bệnh)</summary>
                <div class="detail-content">
                    <p><b>Nguyên tắc FRAC/IRAC:</b> ${protocol.III_Chiến_lược_Kiểm_soát_Hóa_học?.Nguyên_tắc_FRAC_IRAC || '...'}</p>
                    <h4>Phác đồ xử lý:</h4>
                    ${chemicalStepsHtml}
                    <p style="margin-top:10px;"><b>Thuốc trừ tận gốc:</b> ${protocol.III_Chiến_lược_Kiểm_soát_Hóa_học?.Thuốc_Trừ_Tận_gốc_Eradicant || '...'}</p>
                </div>
            </details>
            
            <details class="protocol-detail-section">
                <summary><i class="material-icons">eco</i> IV. Giải pháp Sinh học & Thay thế</summary>
                <div class="detail-content">${bioHtml}</div>
            </details>

            <details class="protocol-detail-section">
                <summary><i class="material-icons">link</i> V. Nguồn thông tin tham khảo</summary>
                <div class="detail-content">${sourcesHtml}</div>
            </details>
        </div>
    `;

    resultContainer.innerHTML = html;
}

// --- PREDICTION LOGIC ---

async function predict(imageElement) {
    if (!model) {
        alert("Mô hình chưa tải xong. Vui lòng đợi giây lát.");
        return;
    }
    
    // UI Reset
    resultContainer.style.display = 'none';
    boxResult.style.display = 'flex'; 
    loadingPredictionBar.style.display = 'flex'; 
    progressBar.set(0);
    confidenceSpan.textContent = 0;
    predClassSpan.textContent = 'Đang phân tích...';
    
    try {
        // Preprocessing: Resize 224x224, Float32, Normalize / 255.0, Expand Dims
        const tensor = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([224, 224]) 
            .toFloat()
            .div(tf.scalar(255.0)) 
            .expandDims(); 

        // Inference
        const predictions = model.predict(tensor);
        const predictionArray = await predictions.data();
        
        // Find max
        const highestPrediction = Math.max(...predictionArray);
        const predicted_index = predictionArray.indexOf(highestPrediction).toString(); 
        const confidence_score = Math.floor(highestPrediction * 100);

        // Cleanup tensor memory
        tensor.dispose(); 
        predictions.dispose();
        
        // Animation update
        loadingPredictionBar.style.display = 'none';
        let normalizedConfidence = confidence_score / 100;
        progressBar.animate(normalizedConfidence, () => {
            confidenceSpan.textContent = confidence_score;
        });

        // Get Name
        const diseaseName = class_indices[predicted_index] || `Không xác định (ID: ${predicted_index})`;
        predClassSpan.textContent = diseaseName;
        
        console.log(`Prediction: ${diseaseName} (${confidence_score}%)`); 
        
        // Fetch Protocol
        const protocol = disease_protocols_map[predicted_index];
        if (protocol) {
            displayDiseaseDetails(protocol);
        } else {
            resultContainer.innerHTML = `<div class="protocol-header"><h3 style="color:orange">Không tìm thấy dữ liệu chi tiết cho bệnh này.</h3></div>`;
            resultContainer.style.display = 'block';
        }

    } catch (e) {
        console.error("Lỗi dự đoán:", e);
        loadingPredictionBar.style.display = 'none';
        predClassSpan.textContent = 'Lỗi!';
        alert("Đã xảy ra lỗi khi xử lý ảnh. Xem console để biết chi tiết.");
    }
}

// --- CAMERA HANDLERS ---

async function startCamera() {
    // Ẩn các phần hiển thị ảnh tĩnh
    boxResult.style.display = 'none';
    resultContainer.style.display = 'none';
    img.style.display = 'none';
    document.querySelector('.image-placeholder').style.display = 'none';
    
    cameraContainer.style.display = 'block';

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            // Ưu tiên camera sau (environment)
            currentStream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'environment' } 
            });
            videoStream.srcObject = currentStream;
            videoStream.style.display = 'block';
            captureButton.style.display = 'flex';
            captureButton.disabled = false;
            stopButton.style.display = 'flex';
            cameraStatus.textContent = 'Camera đang chạy...';
        } catch (error) {
            cameraStatus.textContent = `Không thể truy cập camera: ${error.message}`;
        }
    } else {
        cameraStatus.textContent = 'Trình duyệt không hỗ trợ camera.';
    }
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    videoStream.srcObject = null;
    videoStream.style.display = 'none';
    captureButton.style.display = 'none';
    stopButton.style.display = 'none';
    cameraContainer.style.display = 'none';
    
    // Nếu chưa có ảnh nào được chụp, hiện lại placeholder
    if (img.style.display === 'none') {
        document.querySelector('.image-placeholder').style.display = 'block';
    }
}

// --- EVENT LISTENERS ---

fileUpload.addEventListener('change', function () {
    const file = this.files[0];
    if (file) {
        stopCamera();
        const reader = new FileReader();
        reader.onload = function (e) {
            img.src = e.target.result;
            img.style.display = 'block'; 
            document.querySelector('.image-placeholder').style.display = 'none';
            // Đợi ảnh load xong mới predict
            img.onload = () => predict(img);
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

stopButton.addEventListener('click', stopCamera);

captureButton.addEventListener('click', () => {
    // Vẽ frame hiện tại từ video lên canvas
    context.drawImage(videoStream, 0, 0, canvas.width, canvas.height); 
    const dataUrl = canvas.toDataURL('image/png');
    
    img.src = dataUrl;
    img.style.display = 'block'; 
    document.querySelector('.image-placeholder').style.display = 'none';

    stopCamera(); 
    predict(img);
});

modeToggle.addEventListener('click', () => {
    if (body.classList.contains('light-mode')) {
        body.classList.replace('light-mode', 'dark-mode');
        modeToggle.innerHTML = '<i class="material-icons">wb_sunny</i> Chế độ Sáng';
        progressBar.options.trailColor = '#333333';
    } else {
        body.classList.replace('dark-mode', 'light-mode');
        modeToggle.innerHTML = '<i class="material-icons">brightness_4</i> Chế độ Tối';
        progressBar.options.trailColor = '#e0e0e0';
    }
    // Redraw progress bar trail
    progressBar.set(progressBar.value()); 
});

// Start app
document.addEventListener('DOMContentLoaded', initialize);