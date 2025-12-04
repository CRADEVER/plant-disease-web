let model;
let disease_protocols_map = {}; 
let class_indices = {}; 
let currentStream;

// DOM Elements
const fileUpload = document.getElementById('uploadImage');
const img = document.getElementById('image');
const imagePlaceholder = document.querySelector('.image-placeholder');
const boxResult = document.getElementById('boxResult');
const predClassSpan = document.querySelector('.pred_class');
const confidenceSpan = document.querySelector('.confidence');
const resultContainer = document.getElementById('resultContainer'); 
const mainStatus = document.getElementById('mainStatus'); 
const loadingPredictionBar = document.getElementById('loadingPredictionBar'); 

// Camera Elements
const cameraToggle = document.getElementById('cameraToggle');
const cameraContainer = document.getElementById('cameraContainer');
const videoStream = document.getElementById('videoStream');
const captureButton = document.getElementById('captureButton');
const stopButton = document.getElementById('stopButton');
const cameraStatus = document.getElementById('cameraStatus');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

// Mode Toggle
const modeToggle = document.getElementById('modeToggle');
const body = document.body;

// Progress Bar
const progressBar = new ProgressBar.Circle('#progress', {
    color: '#00a896', 
    strokeWidth: 10,
    duration: 1000,
    easing: 'easeInOut',
    trailColor: '#e0e0e0', 
    trailWidth: 4,
    text: { autoStyleContainer: false }
});

// 1. Tải dữ liệu từ JSON
async function fetchData(){
    try {
        let response = await fetch('./class_indices.json');
        
        if (!response.ok) {
            throw new Error(`HTTP Error! Status: ${response.status}`);
        }
        
        let data = await response.json();
        
        const protocolsArray = data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet;

        if (Array.isArray(protocolsArray)) {
            protocolsArray.forEach(item => {
                // Map ID (string) sang object
                disease_protocols_map[item.Mã_ID] = item;
                // Tạo danh sách tên để hiển thị nhanh
                class_indices[item.Mã_ID] = item.Tên_Bệnh; 
            });
            console.log("DEBUG: Đã tải dữ liệu bệnh thành công.");
        } else {
             throw new Error("JSON không đúng cấu trúc mảng 'Phac_do_Quan_Ly_Tong_Hop_Chi_tiet'");
        }

    } catch (error) {
        console.error("Lỗi tải JSON:", error);
        mainStatus.className = 'status error';
        mainStatus.innerHTML = `<i class="material-icons">error</i> Lỗi dữ liệu: ${error.message}`;
    }
}

// 2. Khởi tạo & Tải Model (ĐÃ FIX LỖI InputLayer)
async function initialize() {
    mainStatus.className = 'status loading';
    mainStatus.innerHTML = '<i class="material-icons loading-icon">cached</i> Đang tải mô hình & dữ liệu...';

    await fetchData();
  
    try {
        const modelUrl = './tensorflowjs-model/model.json'; 
        
        // 1. Tải model dưới dạng LayersModel
        model = await tf.loadLayersModel(modelUrl); 

        // 2. KHẮC PHỤC LỖI INPUTLAYER BẰNG DỰ ĐOÁN GIẢ (DUMMY PREDICTION)
        // Kích thước đầu vào của MobileNetV2 là [224, 224, 3]
        const dummyInput = tf.zeros([1, 224, 224, 3]);
        const output = model.predict(dummyInput);
        
        // Dọn dẹp tensor
        output.dispose();
        dummyInput.dispose();

        mainStatus.className = 'status success';
        mainStatus.innerHTML = '<i class="material-icons">check_circle</i> Hệ thống sẵn sàng.';
    } catch (error) {
        console.error("Lỗi tải Model:", error);
        mainStatus.className = 'status error';
        mainStatus.innerHTML = `<i class="material-icons">error</i> Lỗi tải Model: ${error.message}. Vui lòng kiểm tra file model.json trong thư mục tensorflowjs-model.`;
    }
}

// 3. Hiển thị chi tiết phác đồ
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

    // --- Section III: Hóa học (Xử lý mảng phác đồ) ---
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
        stagesHtml = '<p>Không có phác đồ hóa học cụ thể cho bệnh này.</p>';
    }

    const htmlSec3 = `
        <div class="detail-content">
            <p><b>Nguyên tắc:</b> ${getText(sec3.Nguyên_tắc_FRAC_IRAC)}</p>
            ${stagesHtml}
            <div class="alert-box">
                <b>Thuốc Trừ Tận gốc (Eradicant):</b> ${getText(sec3.Thuốc_Trừ_Tận_gốc_Eradicant)}
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
        // Xử lý cả hai trường hợp: object có URL hoặc chỉ là string URL
        let url = '';
        let name = `Nguồn tham khảo ${key}`;

        if (typeof src === 'object' && src.URL) {
             url = src.URL;
             name = src.Tên_Nguồn || src.URL;
        } else if (typeof src === 'string' && src.startsWith('http')) {
             url = src;
             name = `Nguồn tham khảo ${key}`;
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
    resultContainer.scrollIntoView({ behavior: 'smooth' });
}

// 4. Dự đoán
async function predict(imageElement) {
    if (!model) return;
    
    // UI Updates
    resultContainer.style.display = 'none';
    boxResult.style.display = 'flex'; 
    loadingPredictionBar.style.display = 'flex'; 
    progressBar.set(0);
    confidenceSpan.textContent = 0;
    predClassSpan.textContent = 'Đang phân tích...';
    
    try {
        // Preprocess Image: Resize 224x224, Float, Normalize / 255
        const tensor = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([224, 224]) 
            .toFloat()
            .div(tf.scalar(255.0)) 
            .expandDims(); 

        const predictions = model.predict(tensor);
        const data = await predictions.data();
        
        const maxVal = Math.max(...data);
        const index = data.indexOf(maxVal); 
        const confidence = Math.floor(maxVal * 100);

        // Cleanup tensor
        tensor.dispose(); 
        predictions.dispose();

        // UI Results
        loadingPredictionBar.style.display = 'none';
        
        // Animation Progress
        progressBar.animate(confidence / 100, () => {
            confidenceSpan.textContent = confidence;
        });

        const idString = index.toString();
        const diseaseName = class_indices[idString] || "Không xác định";
        predClassSpan.textContent = diseaseName;
        
        console.log(`Kết quả: ID=${idString}, Tên=${diseaseName}, Độ tin cậy=${confidence}%`);

        // Tìm phác đồ trong map
        const protocol = disease_protocols_map[idString];
        if (protocol) {
            displayDiseaseDetails(protocol);
        } else {
            resultContainer.innerHTML = `<div class="alert-box">Không tìm thấy phác đồ cho ID: ${idString}</div>`;
            resultContainer.style.display = 'block';
        }

    } catch (e) {
        console.error(e);
        loadingPredictionBar.style.display = 'none';
        predClassSpan.textContent = "Lỗi xử lý";
    }
}

// 5. Camera & Event Listeners
async function startCamera() {
    imagePlaceholder.style.display = 'none';
    img.style.display = 'none';
    cameraContainer.style.display = 'block';
    resultContainer.style.display = 'none';
    boxResult.style.display = 'none';

    try {
        // Request camera access, prioritize rear camera ('environment')
        currentStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        videoStream.srcObject = currentStream;
        captureButton.disabled = false;
        cameraStatus.textContent = '';
    } catch (error) {
        cameraStatus.textContent = 'Không thể truy cập camera. Vui lòng kiểm tra quyền truy cập.';
        console.error("Camera error:", error);
    }
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(t => t.stop());
    }
    cameraContainer.style.display = 'none';
    imagePlaceholder.style.display = 'block';
    captureButton.disabled = true;
    cameraStatus.textContent = '';
}

fileUpload.addEventListener('change', function () {
    const file = this.files[0];
    if (file) {
        stopCamera();
        const reader = new FileReader();
        reader.onload = function (e) {
            img.src = e.target.result;
            img.style.display = 'block'; 
            imagePlaceholder.style.display = 'none';
            // Đợi ảnh load xong mới predict
            img.onload = () => predict(img);
        };
        reader.readAsDataURL(file);
    }
});

cameraToggle.addEventListener('click', () => {
    if (cameraContainer.style.display === 'block') stopCamera();
    else startCamera();
});

captureButton.addEventListener('click', () => {
    // Đảm bảo canvas có kích thước thực của video để chụp ảnh chất lượng cao
    canvas.width = videoStream.videoWidth;
    canvas.height = videoStream.videoHeight;
    context.drawImage(videoStream, 0, 0, canvas.width, canvas.height);
    
    // Gán ảnh đã chụp vào thẻ <img>
    img.src = canvas.toDataURL('image/png');
    img.style.display = 'block';
    
    stopCamera();
    // Chờ 1 chút để UI cập nhật rồi predict
    setTimeout(() => predict(img), 100);
});

stopButton.addEventListener('click', stopCamera);

modeToggle.addEventListener('click', () => {
    body.classList.toggle('dark-mode');
    body.classList.toggle('light-mode');
    const isDark = body.classList.contains('dark-mode');
    modeToggle.innerHTML = isDark ? 
        '<i class="material-icons">wb_sunny</i> Chế độ Sáng' : 
        '<i class="material-icons">brightness_4</i> Chế độ Tối';
});

// Run Init
document.addEventListener('DOMContentLoaded', initialize);