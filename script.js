let model;
let disease_protocols_map = {}; 
let class_indices = {}; 
let currentStream;

// ==============================================================
// CẤU HÌNH ĐƯỜNG DẪN
// Nếu chạy local (Live Server), để trống "".
// Nếu chạy GitHub Pages: "/ten-repo/"
const BASE_PATH = ""; 
// ==============================================================

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('uploadImage');
const imgElement = document.getElementById('image');
const imagePlaceholder = document.getElementById('imagePlaceholder');
const scanOverlay = document.getElementById('scanOverlay');
const boxResult = document.getElementById('boxResult');
const predClassSpan = document.querySelector('.pred_class');
const confidenceSpan = document.querySelector('.confidence');
const resultContainer = document.getElementById('resultContainer'); 
const mainStatus = document.getElementById('mainStatus'); 

// Camera Elements
const cameraToggle = document.getElementById('cameraToggle');
const cameraContainer = document.getElementById('cameraContainer');
const videoStream = document.getElementById('videoStream');
const captureButton = document.getElementById('captureButton');
const stopButton = document.getElementById('stopButton');
const cameraStatus = document.getElementById('cameraStatus');

// Mode Toggle
const modeToggle = document.getElementById('modeToggle');
const body = document.body;

// Init Progress Bar
const progressBar = new ProgressBar.Circle('#progress', {
    color: '#00a896', 
    strokeWidth: 8,
    trailWidth: 4,
    easing: 'easeInOut',
    duration: 1400,
    text: { autoStyleContainer: false },
    from: { color: '#eee', width: 4 },
    to: { color: '#00a896', width: 8 },
    step: function(state, circle) {
        circle.path.setAttribute('stroke', state.color);
        circle.path.setAttribute('stroke-width', state.width);
        circle.setText(''); // Text được xử lý riêng
    }
});

// 1. Tải dữ liệu JSON
async function fetchData() {
    try {
        let response = await fetch(`${BASE_PATH}class_indices.json`);
        if (!response.ok) throw new Error("Không thể tải file dữ liệu bệnh (class_indices.json)");
        let data = await response.json();
        
        const protocolsArray = data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet;
        if (Array.isArray(protocolsArray)) {
            protocolsArray.forEach(item => {
                disease_protocols_map[item.Mã_ID] = item;
                class_indices[item.Mã_ID] = item.Tên_Bệnh; 
            });
            console.log("✅ Dữ liệu bệnh đã tải.");
        }
    } catch (error) {
        console.error(error);
        mainStatus.className = 'status error';
        mainStatus.innerHTML = `<i class="material-icons">error</i> Lỗi dữ liệu: ${error.message}`;
    }
}

// 2. Khởi tạo & Load Model
async function initialize() {
    mainStatus.className = 'status loading';
    mainStatus.innerHTML = '<i class="material-icons loading-icon">cached</i> Đang tải mô hình & dữ liệu...';
    
    await fetchData();

    try {
        const modelUrl = `${BASE_PATH}tensorflowjs-model/model.json`;
        model = await tf.loadLayersModel(modelUrl);
        
        // Warmup (Chạy thử 1 lần để model sẵn sàng)
        tf.tidy(() => {
            model.predict(tf.zeros([1, 224, 224, 3]));
        });

        mainStatus.className = 'status success';
        mainStatus.innerHTML = '<i class="material-icons">check_circle</i> Hệ thống sẵn sàng';
        
        // Kích hoạt UI
        dropZone.style.pointerEvents = 'auto';
        cameraToggle.disabled = false;

    } catch (error) {
        console.error(error);
        mainStatus.className = 'status error';
        mainStatus.innerHTML = `<i class="material-icons">error</i> Không tải được Model. Kiểm tra đường dẫn: ${BASE_PATH}`;
    }
}

// 3. Xử lý ảnh và Dự đoán (Tối ưu hóa bộ nhớ)
async function processImageAndPredict(imageSource) {
    if (!model) return;

    // Reset UI
    boxResult.style.display = 'flex';
    resultContainer.style.display = 'none';
    progressBar.set(0);
    confidenceSpan.textContent = "0";
    predClassSpan.textContent = "Đang phân tích...";
    
    // Hiệu ứng scan
    imgElement.parentElement.classList.add('scanning');

    try {
        // Tối ưu: Vẽ ảnh lên Canvas ẩn 224x224 trước khi chuyển thành Tensor
        // Giúp tránh lỗi bộ nhớ khi ảnh đầu vào quá lớn (ví dụ ảnh từ iPhone 4K)
        const offScreenCanvas = document.createElement('canvas');
        offScreenCanvas.width = 224;
        offScreenCanvas.height = 224;
        const ctx = offScreenCanvas.getContext('2d');
        ctx.drawImage(imageSource, 0, 0, 224, 224);

        // Chuyển canvas thành tensor và chuẩn hóa
        const tensor = tf.tidy(() => {
            return tf.browser.fromPixels(offScreenCanvas)
                .toFloat()
                .div(tf.scalar(255.0))
                .expandDims();
        });

        // Delay 500ms để người dùng thấy hiệu ứng scan
        await new Promise(r => setTimeout(r, 500));

        const predictions = await model.predict(tensor).data();
        const maxVal = Math.max(...predictions);
        const index = predictions.indexOf(maxVal);
        
        // Dọn dẹp bộ nhớ tensor
        tensor.dispose();
        imgElement.parentElement.classList.remove('scanning');

        // Hiển thị kết quả
        const confidence = Math.floor(maxVal * 100);
        progressBar.animate(confidence / 100);
        confidenceSpan.textContent = confidence;

        const idString = index.toString();
        const diseaseName = class_indices[idString] || `Bệnh chưa xác định (ID: ${idString})`;
        predClassSpan.textContent = diseaseName;

        // Tìm và hiển thị phác đồ
        const protocol = disease_protocols_map[idString];
        if (protocol) {
            displayDiseaseDetails(protocol);
        } else {
            resultContainer.innerHTML = `<div class="alert-box">Không tìm thấy phác đồ chi tiết cho bệnh này.</div>`;
            resultContainer.style.display = 'block';
        }

    } catch (e) {
        console.error(e);
        imgElement.parentElement.classList.remove('scanning');
        predClassSpan.textContent = "Lỗi xử lý ảnh";
    }
}

// 4. Hiển thị chi tiết phác đồ (Logic Render HTML)
function displayDiseaseDetails(protocol) {
    resultContainer.style.display = 'block';

    const getText = (val) => val ? val : 'Đang cập nhật...';

    // --- Section I: Tác nhân & Điều kiện ---
    const sec1 = protocol.I_Tác_nhân_Chu_kỳ_và_Điều_kiện || {};
    const htmlSec1 = `
        <div class="detail-content">
            <p><b>Tác nhân:</b> ${getText(sec1.Tác_nhân_Sinh_học)}</p>
            <p><b>Lây lan:</b> ${getText(sec1.Cơ_chế_Lây_lan)}</p>
            <p><b>Điều kiện tối ưu:</b> ${getText(sec1.Nhiệt_độ_Thời_điểm_tối_ưu)}</p>
            <div class="highlight-box">
                <b>Dấu hiệu nhận biết:</b> <br>
                ${getText(sec1.Dấu_hiệu_Chẩn_đoán_Chuyên_sâu).replace(/\n/g, '<br>')}
            </div>
        </div>
    `;

    // --- Section II: Canh tác ---
    const sec2 = protocol.II_Biện_pháp_Canh_tác_Chuyên_sâu || {};
    const htmlSec2 = `
        <div class="detail-content">
            <ul class="custom-list">
                <li><b>Đất & Tàn dư:</b> ${getText(sec2.Quản_lý_Tàn_dư_Đất)}</li>
                <li><b>Nước tưới:</b> ${getText(sec2.Quản_lý_Nước_Tưới)}</li>
                <li><b>Mật độ:</b> ${getText(sec2.Mật_độ_Thông_thoáng)}</li>
                <li><b>Dinh dưỡng:</b> ${getText(sec2.Quản_lý_Dinh_dưỡng_Vi_lượng)}</li>
            </ul>
        </div>
    `;

    // --- Section III: Hóa học ---
    const sec3 = protocol.III_Chiến_lược_Kiểm_soát_Hóa_học || {};
    const stagesArr = sec3.Phác_đồ_Giai_đoạn_Cây || [];
    
    let stagesHtml = '';
    if (stagesArr.length > 0) {
        stagesHtml = stagesArr.map(step => `
            <div class="stage-step">
                <div class="step-header"><i class="material-icons">event</i> ${step.Giai_đoạn || 'Giai đoạn'}</div>
                <p><b>Hoạt chất:</b> <span class="chem-name">${step.Hoạt_chất_Đề_xuất}</span></p>
                <p><b>Nhóm FRAC/IRAC:</b> ${step.Nhóm_FRAC_IRAC}</p>
                <p><i>Lưu ý: ${step.Lưu_ý_Ứng_dụng}</i></p>
            </div>
        `).join('');
    } else {
        stagesHtml = '<p>Không có phác đồ hóa học cụ thể cho bệnh này hoặc không khuyến cáo dùng thuốc.</p>';
    }

    const htmlSec3 = `
        <div class="detail-content">
            <p><b>Nguyên tắc:</b> ${getText(sec3.Nguyên_tắc_FRAC_IRAC)}</p>
            ${stagesHtml}
            <div class="alert-box">
                <b>Thuốc đặc trị (Eradicant):</b> ${getText(sec3.Thuốc_Trừ_Tận_gốc_Eradicant)}
            </div>
        </div>
    `;

    // --- Section IV: Sinh học ---
    const sec4 = protocol.IV_Giải_pháp_Sinh_học_và_Thay_thế || {};
    const htmlSec4 = `
        <div class="detail-content">
            <p><b>Chất đối kháng:</b> ${getText(sec4.Chất_Đối_kháng_VSV)}</p>
            <p><b>Kích kháng (SAR):</b> ${getText(sec4.Cảm_ứng_Kháng_Bệnh_SAR)}</p>
            <p><b>Kiểm soát Vector:</b> ${getText(sec4.Kiểm_soát_Côn_trùng_Vector)}</p>
        </div>
    `;

    // --- Section V: Nguồn tham khảo ---
    const sec5 = protocol.V_Nguồn_Thông_Tin || {};
    let sourcesHtml = '';
    Object.keys(sec5).forEach(key => {
        const src = sec5[key];
        let url = '', name = `Nguồn tham khảo ${key}`;

        if (typeof src === 'object' && src.URL) {
             url = src.URL;
             name = src.Tên_Nguồn || src.URL;
        } else if (typeof src === 'string' && src.startsWith('http')) {
             url = src;
        }
        
        if (url) {
            sourcesHtml += `<li><a href="${url}" target="_blank" class="source-link">${name}</a></li>`;
        }
    });

    const finalHtml = `
        <div class="protocol-header">
            <h3>${protocol.Tên_Bệnh}</h3>
            <span class="badge">${protocol.Phân_loại}</span>
        </div>
        
        <div class="protocol-sections">
            <details class="protocol-detail-section" open>
                <summary><i class="material-icons">bug_report</i> I. Tác nhân & Điều kiện</summary>
                ${htmlSec1}
            </details>

            <details class="protocol-detail-section">
                <summary><i class="material-icons">agriculture</i> II. Biện pháp Canh tác</summary>
                ${htmlSec2}
            </details>

            <details class="protocol-detail-section" open>
                <summary><i class="material-icons">science</i> III. Kiểm soát Hóa học</summary>
                ${htmlSec3}
            </details>
            
            <details class="protocol-detail-section">
                <summary><i class="material-icons">eco</i> IV. Giải pháp Sinh học</summary>
                ${htmlSec4}
            </details>

            ${sourcesHtml ? `
            <div class="sources-section">
                <h4><i class="material-icons">link</i> Nguồn thông tin xác thực:</h4>
                <ul>${sourcesHtml}</ul>
            </div>` : ''}
        </div>
    `;

    resultContainer.innerHTML = finalHtml;
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// 5. Event Listeners (Xử lý sự kiện)

// -- Upload File --
fileInput.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));

// -- Drag & Drop --
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});
function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

dropZone.addEventListener('dragover', () => dropZone.classList.add('drag-over'));
['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'));
});
dropZone.addEventListener('drop', (e) => handleFileSelect(e.dataTransfer.files[0]));

function handleFileSelect(file) {
    if (file && file.type.startsWith('image/')) {
        stopCamera();
        const reader = new FileReader();
        reader.onload = (e) => {
            imgElement.src = e.target.result;
            imgElement.style.display = 'block';
            imagePlaceholder.style.display = 'none';
            imgElement.onload = () => processImageAndPredict(imgElement);
        };
        reader.readAsDataURL(file);
    }
}

// -- Camera Logic --
async function startCamera() {
    imagePlaceholder.style.display = 'none';
    imgElement.style.display = 'none';
    cameraContainer.style.display = 'block';
    boxResult.style.display = 'none';
    resultContainer.style.display = 'none';

    try {
        currentStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        videoStream.srcObject = currentStream;
        captureButton.disabled = false;
        cameraStatus.textContent = "";
    } catch (err) {
        cameraStatus.textContent = "Không thể mở Camera: " + err.message;
    }
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
    cameraContainer.style.display = 'none';
    captureButton.disabled = true;
    
    if (imgElement.src === "" || imgElement.style.display === 'none') {
        imagePlaceholder.style.display = 'flex';
    }
}

cameraToggle.addEventListener('click', () => {
    if (cameraContainer.style.display === 'block') stopCamera();
    else startCamera();
});

captureButton.addEventListener('click', () => {
    const canvas = document.createElement('canvas');
    canvas.width = videoStream.videoWidth;
    canvas.height = videoStream.videoHeight;
    canvas.getContext('2d').drawImage(videoStream, 0, 0);
    
    imgElement.src = canvas.toDataURL('image/png');
    imgElement.style.display = 'block';
    
    stopCamera();
    setTimeout(() => processImageAndPredict(imgElement), 100);
});

stopButton.addEventListener('click', stopCamera);

// -- Dark Mode --
modeToggle.addEventListener('click', () => {
    body.classList.toggle('dark-mode');
    body.classList.toggle('light-mode');
    const isDark = body.classList.contains('dark-mode');
    modeToggle.querySelector('span').textContent = isDark ? "Chế độ Sáng" : "Chế độ Tối";
    modeToggle.querySelector('i').textContent = isDark ? "wb_sunny" : "brightness_4";
});

// Chạy Init khi trang tải xong
document.addEventListener('DOMContentLoaded', initialize);