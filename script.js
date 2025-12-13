/* =========================================
   IPDM SYSTEM - LOGIC CORE (Phiên bản tối ưu)
   ========================================= */

// --- Biến toàn cục ---
let model;
let disease_protocols_map = {}; 
let class_indices = {}; 
let currentStream = null;

// --- DOM Elements ---
const fileUpload = document.getElementById('uploadImage');
const imgElement = document.getElementById('image');
const placeholder = document.querySelector('.image-placeholder');
const boxResult = document.getElementById('boxResult');
const predClassSpan = document.querySelector('.pred_class');
const confidenceSpan = document.querySelector('.confidence');
const resultContainer = document.getElementById('resultContainer'); 
const mainStatus = document.getElementById('mainStatus'); 
const loadingBar = document.getElementById('loadingPredictionBar'); 

// Camera Elements
const cameraToggle = document.getElementById('cameraToggle');
const cameraContainer = document.getElementById('cameraContainer');
const videoStream = document.getElementById('videoStream');
const captureButton = document.getElementById('captureButton');
const stopButton = document.getElementById('stopButton');
const cameraStatus = document.getElementById('cameraStatus');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

// UI Elements
const modeToggle = document.getElementById('modeToggle');
const body = document.body;

// Progress Bar Init
const progressBar = new ProgressBar.Circle('#progress', {
    color: '#00a896', 
    strokeWidth: 8,
    trailWidth: 4,
    easing: 'easeInOut',
    duration: 1000, // Rút ngắn duration cho phản hồi nhanh hơn
    text: { autoStyleContainer: false },
    from: { color: '#aaa', width: 4 },
    to: { color: '#00a896', width: 8 },
    step: function(state, circle) {
        circle.path.setAttribute('stroke', state.color);
        circle.path.setAttribute('stroke-width', state.width);
        const value = Math.round(circle.value() * 100);
        circle.setText(value + '%'); // Luôn hiển thị phần trăm (cũ: không hiển thị 0%)
    }
});
// Cài đặt style cho text trong Progress Bar
progressBar.text.style.fontFamily = 'Roboto, sans-serif'; 
progressBar.text.style.fontSize = '2rem';
progressBar.text.style.fontWeight = '700';
// Cập nhật màu chữ dựa trên mode
function updateProgressBarTextColor() {
    progressBar.text.style.color = body.classList.contains('dark-mode') ? '#ccc' : '#555';
}

// --- HÀM KHỞI TẠO DỮ LIỆU ---
async function fetchData() {
    mainStatus.className = 'status loading';
    mainStatus.innerHTML = '<i class="material-icons loading-icon">autorenew</i> Đang tải dữ liệu phác đồ...';
    try {
        let response = await fetch('./class_indices.json');
        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
        
        let data = await response.json();
        
        // Mapping dữ liệu từ JSON
        const protocolsArray = data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet;
        if (Array.isArray(protocolsArray)) {
            protocolsArray.forEach(item => {
                // Map ID (STRING) sang object dữ liệu
                disease_protocols_map[item.Mã_ID] = item;
                // Map ID (STRING) sang Tên bệnh
                class_indices[item.Mã_ID] = item.Tên_Bệnh; 
            });
            console.log("Dữ liệu phác đồ đã tải: ", Object.keys(disease_protocols_map).length, "bệnh.");
        } else {
             throw new Error("JSON sai cấu trúc: Không tìm thấy mảng 'Phac_do_Quan_Ly_Tong_Hop_Chi_tiet'.");
        }
        mainStatus.className = 'status loading';
        mainStatus.innerHTML = '<i class="material-icons loading-icon">autorenew</i> Đang tải mô hình AI...';

    } catch (error) {
        console.error("Load Data Error:", error);
        mainStatus.className = 'status error';
        mainStatus.innerHTML = `<i class="material-icons">error</i> Lỗi tải dữ liệu: ${error.message}`;
    }
}

async function loadModel() {
    try {
        // Fix: Sử dụng loadGraphModel nếu model được convert từ SavedModel (Khuyến nghị)
        // Nếu bạn dùng model convert bằng save_keras_model, hãy đổi lại là tf.loadLayersModel
        model = await tf.loadGraphModel('./tensorflowjs-model/model.json'); 
        
        // Warm-up model (chạy thử 1 lần)
        const dummy = tf.zeros([1, 224, 224, 3]);
        model.predict(dummy).dispose();
        dummy.dispose();

        mainStatus.className = 'status success';
        mainStatus.innerHTML = '<i class="material-icons">check_circle</i> Hệ thống AI sẵn sàng. Vui lòng tải ảnh.';
        
        updateProgressBarTextColor(); // Khởi tạo màu cho Progress Bar

    } catch (error) {
        console.error("Model Error:", error);
        mainStatus.className = 'status error';
        mainStatus.innerHTML = '<i class="material-icons">broken_image</i> Lỗi tải mô hình AI. Kiểm tra đường dẫn model và xem đã đổi thành loadGraphModel chưa.';
    }
}

async function initialize() {
    await fetchData();
    await loadModel();
}

// --- HÀM HIỂN THỊ PHÁC ĐỒ CHI TIẾT (RENDER JSON) ---
function displayDiseaseDetails(protocol) {
    resultContainer.style.display = 'block';
    
    // Đảm bảo Mã ID là số để so sánh với index
    const isHealthy = protocol.Tên_Bệnh.toLowerCase().includes("khỏe mạnh");

    // 1. Render Phần II: Biện pháp canh tác
    const cult = protocol.II_Biện_pháp_Canh_tác_Chuyên_sâu || {};
    const sectionCultural = `
        <ul class="info-list">
            <li><strong><i class="material-icons tiny">delete</i> Vệ sinh:</strong> ${cult.Quản_lý_Tàn_dư_Đất || 'N/A'}</li>
            <li><strong><i class="material-icons tiny">water_drop</i> Tưới tiêu:</strong> ${cult.Quản_lý_Nước_Tưới || 'N/A'}</li>
            <li><strong><i class="material-icons tiny">grid_on</i> Mật độ:</strong> ${cult.Mật_độ_Thông_thoáng || 'N/A'}</li>
            <li><strong><i class="material-icons tiny">eco</i> Dinh dưỡng:</strong> ${cult.Quản_lý_Dinh_dưỡng_Vi_lượng || 'N/A'}</li>
        </ul>
    `;

    // 2. Render Phần III: Hóa học (Ẩn nếu cây khỏe)
    let sectionChemical = '';
    if (!isHealthy) {
        const chem = protocol.III_Chiến_lược_Kiểm_soát_Hóa_học || {};
        const stages = chem.Phác_đồ_Giai_đoạn_Cây || [];
        
        let stagesHtml = '';
        if (stages.length > 0) {
            stagesHtml = stages.map(step => `
                <div class="stage-step">
                    <div class="step-title"><i class="material-icons">schedule</i> Giai đoạn: ${step.Giai_đoạn}</div>
                    <div class="step-content">
                        <p><strong>Hoạt chất:</strong> <span class="highlight-chem">${step.Hoạt_chất_Đề_xuất}</span></p>
                        <p class="sub-text">Nhóm FRAC/IRAC: ${step.Nhóm_FRAC_IRAC} | Lưu ý: ${step.Lưu_ý_Ứng_dụng}</p>
                    </div>
                </div>
            `).join('');
        } else {
            stagesHtml = '<p><em>Không có phác đồ hóa học cụ thể cho bệnh này.</em></p>';
        }

        sectionChemical = `
            <details class="protocol-detail-section">
                <summary><i class="material-icons">science</i> III. Chiến lược Hóa học (Thuốc BVTV)</summary>
                <div class="detail-content">
                    <p><strong>Nguyên tắc:</strong> ${chem.Nguyên_tắc_FRAC_IRAC || 'N/A'}</p>
                    <div class="stages-container">${stagesHtml}</div>
                    <p class="warning-text"><i class="material-icons">warning</i> <strong>Thuốc trị tận gốc (Eradicant):</strong> ${chem.Thuốc_Trừ_Tận_gốc_Eradicant || 'N/A'}</p>
                </div>
            </details>
        `;
    }

    // 3. Render Phần IV: Sinh học
    const bio = protocol.IV_Giải_pháp_Sinh_học_và_Thay_thế || {};
    const sectionBio = `
        <p><strong><i class="material-icons tiny">bug_report</i> Vi sinh vật đối kháng:</strong> ${bio.Chất_Đối_kháng_VSV || 'N/A'}</p>
        <p><strong><i class="material-icons tiny">shield</i> Kích kháng (SAR):</strong> ${bio.Cảm_ứng_Kháng_Bệnh_SAR || 'N/A'}</p>
        <p><strong><i class="material-icons tiny">pest_control</i> Kiểm soát Vector:</strong> ${bio.Kiểm_soát_Côn_trùng_Vector || 'N/A'}</p>
    `;

    // 4. Render Phần V: Nguồn tham khảo
    let sourcesHtml = '';
    const sources = protocol.V_Nguồn_Thông_Tin;
    if (sources && typeof sources === 'object') {
        sourcesHtml = '<ul class="source-list">';
        Object.values(sources).forEach(src => {
            // Kiểm tra cả Tên_Nguồn và URL
            if (src.URL && src.Tên_Nguồn) {
                sourcesHtml += `<li><a href="${src.URL}" target="_blank" rel="noopener noreferrer"><i class="material-icons tiny">link</i> ${src.Tên_Nguồn}</a></li>`;
            }
        });
        sourcesHtml += '</ul>';
    }

    // --- Lắp ráp HTML ---
    const html = `
        <div class="protocol-header">
            <h3>${protocol.Tên_Bệnh}</h3>
            <span class="badge ${isHealthy ? 'badge-success' : 'badge-warning'}">${protocol.Phân_loại}</span>
        </div>
        
        <div class="protocol-sections">
            <details class="protocol-detail-section" open>
                <summary><i class="material-icons">search</i> I. Chẩn đoán & Tác nhân</summary>
                <div class="detail-content">
                    <p><strong>Tác nhân:</strong> ${protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện?.Tác_nhân_Sinh_học || 'N/A'}</p>
                    <p><strong>Điều kiện lây lan:</strong> ${protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện?.Nhiệt_độ_Thời_điểm_tối_ưu || 'N/A'}</p>
                    <div class="symptom-box">
                        <strong>Dấu hiệu nhận biết:</strong><br>
                        ${protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện?.Dấu_hiệu_Chẩn_đoán_Chuyên_sâu ? protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Dấu_hiệu_Chẩn_đoán_Chuyên_sâu.replace(/\n/g, '<br>') : 'N/A'}
                    </div>
                </div>
            </details>

            <details class="protocol-detail-section">
                <summary><i class="material-icons">agriculture</i> II. Biện pháp Canh tác</summary>
                <div class="detail-content">${sectionCultural}</div>
            </details>

            ${sectionChemical}
            
            <details class="protocol-detail-section">
                <summary><i class="material-icons">spa</i> IV. Giải pháp Sinh học</summary>
                <div class="detail-content">${sectionBio}</div>
            </details>

            <details class="protocol-detail-section">
                <summary><i class="material-icons">library_books</i> V. Tài liệu tham khảo</summary>
                <div class="detail-content">${sourcesHtml || '<p>Không có nguồn tham khảo.</p>'}</div>
            </details>
        </div>
    `;

    resultContainer.innerHTML = html;
    
    // Smooth scroll tới kết quả
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// --- HÀM DỰ ĐOÁN (AI PREDICT) ---
async function predict(imageElement) {
    if (!model) {
        alert("Mô hình chưa tải xong. Vui lòng đợi!");
        return;
    }

    // Reset UI
    resultContainer.style.display = 'none';
    boxResult.style.display = 'flex';
    loadingBar.style.display = 'flex';
    predClassSpan.innerText = "";
    confidenceSpan.innerText = "0";
    progressBar.set(0);

    try {
        // Tiền xử lý ảnh (Preprocessing)
        const tensor = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(tf.scalar(255.0))
            .expandDims();

        // Chạy dự đoán
        const predictions = await model.predict(tensor).data();
        
        // Tìm lớp có xác suất cao nhất
        const maxPrediction = Math.max(...predictions);
        const maxIndex = predictions.indexOf(maxPrediction);
        
        // Clean memory
        tensor.dispose();

        // Hiển thị kết quả
        loadingBar.style.display = 'none';
        
        const confidencePercent = (maxPrediction * 100);
        
        // Animation Progress bar
        progressBar.animate(maxPrediction, {
            duration: 1000
        }, function() {
            confidenceSpan.innerText = confidencePercent.toFixed(2); // Giữ 2 chữ số thập phân
        });

        // FIX: Chuyển maxIndex (number) sang String để lookup an toàn trong map
        const maxIdString = String(maxIndex); 
        const diseaseName = class_indices[maxIdString] || "Không xác định";
        predClassSpan.innerText = diseaseName;
        
        // Lấy phác đồ tương ứng
        const protocol = disease_protocols_map[maxIdString];
        
        if (protocol) {
            displayDiseaseDetails(protocol);
        } else {
            resultContainer.innerHTML = `<div class="alert alert-warning">Không tìm thấy dữ liệu chi tiết cho ID: ${maxIdString}</div>`;
            resultContainer.style.display = 'block';
        }

    } catch (error) {
        console.error("Prediction Error:", error);
        loadingBar.style.display = 'none';
        predClassSpan.innerText = "Lỗi xử lý";
        alert("Đã xảy ra lỗi khi xử lý ảnh. Chi tiết lỗi trong Console.");
    }
}

// --- LOGIC CAMERA & UPLOAD ---

// 1. Xử lý Upload ảnh
fileUpload.addEventListener('change', function (e) {
    const file = this.files[0];
    if (file) {
        stopCamera(); // Tắt camera nếu đang bật
        const reader = new FileReader();
        reader.onload = function (evt) {
            imgElement.src = evt.target.result;
            imgElement.style.display = 'block';
            placeholder.style.display = 'none';
            
            // Chờ ảnh load xong mới predict
            imgElement.onload = () => predict(imgElement);
        };
        reader.readAsDataURL(file);
    }
});

// 2. Bật Camera
async function startCamera() {
    // Ẩn ảnh upload cũ và kết quả
    imgElement.style.display = 'none';
    placeholder.style.display = 'none';
    resultContainer.style.display = 'none';
    boxResult.style.display = 'none';

    cameraContainer.style.display = 'block';
    
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            // Ưu tiên camera sau trên điện thoại ('environment')
            const constraints = { 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                } 
            };
            currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            videoStream.srcObject = currentStream;
            videoStream.style.display = 'block';
            
            // Hiện nút điều khiển
            captureButton.style.display = 'inline-flex';
            stopButton.style.display = 'inline-flex';
            cameraStatus.innerText = "Đã khởi động camera. Hãy chụp ảnh cây trồng.";
            captureButton.removeAttribute('disabled'); // Bỏ disabled khi stream sẵn sàng
            
        } catch (err) {
            console.error(err);
            cameraStatus.innerText = "Không thể truy cập Camera. Hãy kiểm tra quyền truy cập.";
        }
    } else {
        cameraStatus.innerText = "Trình duyệt không hỗ trợ Camera.";
    }
}

// 3. Tắt Camera
function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    videoStream.srcObject = null;
    videoStream.style.display = 'none';
    cameraContainer.style.display = 'none';
    captureButton.style.display = 'none';
    stopButton.style.display = 'none';
    captureButton.setAttribute('disabled', 'true');
    cameraStatus.innerText = "";
}

// Event Listeners Camera
cameraToggle.addEventListener('click', () => {
    if (!currentStream) {
        startCamera();
    } else {
        stopCamera();
    }
});

stopButton.addEventListener('click', stopCamera);

captureButton.addEventListener('click', () => {
    // Vẽ frame hiện tại lên canvas
    canvas.width = videoStream.videoWidth;
    canvas.height = videoStream.videoHeight;
    context.drawImage(videoStream, 0, 0, canvas.width, canvas.height);
    
    // Chuyển thành ảnh hiển thị
    imgElement.src = canvas.toDataURL('image/jpeg', 0.9); // Tối ưu hóa chất lượng ảnh chụp
    imgElement.style.display = 'block';
    placeholder.style.display = 'none';
    
    stopCamera(); // Tắt camera sau khi chụp
    
    // Dự đoán
    predict(imgElement);
});

// --- CHẾ ĐỘ TỐI / SÁNG ---
modeToggle.addEventListener('click', () => {
    if (body.classList.contains('light-mode')) {
        body.classList.replace('light-mode', 'dark-mode');
        modeToggle.innerHTML = '<i class="material-icons">wb_sunny</i> Chế độ Sáng';
    } else {
        body.classList.replace('dark-mode', 'light-mode');
        modeToggle.innerHTML = '<i class="material-icons">brightness_4</i> Chế độ Tối';
    }
    updateProgressBarTextColor(); // Cập nhật màu chữ
});

// --- KHỞI CHẠY ---
document.addEventListener('DOMContentLoaded', initialize);
