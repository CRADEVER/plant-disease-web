/* =========================================
   IPDM SYSTEM - LOGIC CORE (Phiên bản Sửa lỗi Triệt để)
   ========================================= */

// --- Biến toàn cục ---
let model;
let disease_protocols_map = {}; 
let class_indices = {}; 
let currentStream = null;
let isScanning = false;
let scanInterval = null;
let lastPredictionId = null; 

// Đường dẫn Model (Cần thay thế bằng đường dẫn model thực tế)
const MODEL_PATH = './model/model.json';
const THRESHOLD = 0.8; // Ngưỡng dự đoán (80%)

// --- DOM Elements ---
const fileUpload = document.getElementById('uploadImage');
const imgElement = document.getElementById('image'); // img trong image-box
const placeholder = document.querySelector('.image-placeholder');
const boxResult = document.getElementById('boxResult');
const predClassSpan = document.querySelector('.pred_class');
const confidenceSpan = document.querySelector('.confidence');
const mainStatus = document.getElementById('mainStatus'); // Thanh trạng thái chính
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

// Overlay (Detail & Scope)
const detailOverlay = document.getElementById('detailOverlay');
const detailContent = document.getElementById('detailContent');
const closeDetailBtn = document.getElementById('closeDetailBtn');
const scopeOverlay = document.getElementById('scopeOverlay');
const scopeContent = document.getElementById('scopeContent');
const closeScopeBtn = document.getElementById('closeScopeBtn');
const scopeBtn = document.getElementById('scopeBtn'); // Nút mở Phạm vi

// Progress Bar Init (Tái tạo lại vì bị thiếu trong code cũ)
let progressBar;
$(document).ready(function() {
    progressBar = new ProgressBar.Circle('#progress', {
        color: '#00a896',
        strokeWidth: 10,
        trailWidth: 10,
        trailColor: '#f3f3f3',
        easing: 'easeInOut',
        duration: 1400,
        text: {
            autoStyleContainer: false
        },
        from: { color: '#FFEA82', a: 0 },
        to: { color: '#00a896', a: 1 },
        step: function(state, circle) {
            circle.path.setAttribute('stroke', state.color);
        }
    });

    // Kích hoạt hiệu ứng mặt nước trên background
    try {
        // Cần đảm bảo thư viện ripples đã được link trong index.html
        $('#ripple-background').ripples({ 
            resolution: 512, 
            dropRadius: 20, 
            perturbance: 0.04,
            interactive: true
        });
        console.log("Hiệu ứng mặt nước đã kích hoạt.");
    } catch (e) {
        console.error("Lỗi kích hoạt hiệu ứng Ripples. Đã cài jQuery và Ripples chưa?", e);
    }
    
    initialize();
});

// --- 1. KHỞI TẠO HỆ THỐNG ---

async function initialize() {
    mainStatus.innerText = "Đang tải dữ liệu bệnh...";
    try {
        await fetchData();
        mainStatus.innerText = "Đang tải mô hình AI...";
        await loadModel();
        
        mainStatus.innerText = "Hệ thống SẴN SÀNG";
        mainStatus.style.backgroundColor = "#00a896";
        
    } catch (e) {
        console.error("Lỗi khởi tạo hệ thống:", e);
        mainStatus.innerText = "Lỗi nghiêm trọng: Không thể khởi tạo hệ thống.";
        mainStatus.style.backgroundColor = "#f44336";
    }
}

// --- SỬA LỖI TẢI DỮ LIỆU ---
async function fetchData() {
    try {
        let response = await fetch('./class_indices.json');
        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
        let data = await response.json();
        
        // Lấy mảng dữ liệu bệnh từ khóa chính
        const protocolsArray = data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet; 
        
        if (Array.isArray(protocolsArray) && protocolsArray.length > 0) {
            protocolsArray.forEach(item => {
                // Sử dụng Ma_ID làm key
                const key = String(item.Ma_ID); 
                disease_protocols_map[key] = item;
                class_indices[key] = item.Ten_Benh_Tieng_Viet; // Lưu tên bệnh vào index
            });
            console.log(`Dữ liệu đã tải thành công: ${Object.keys(disease_protocols_map).length} mục.`);
        } else {
             console.error("JSON Error: Mảng dữ liệu rỗng hoặc không đúng định dạng.");
             throw new Error("Dữ liệu JSON rỗng.");
        }
    } catch (error) {
        console.error("Data Error:", error);
        throw error;
    }
}

async function loadModel() {
    // Tải mô hình đã được chuyển đổi từ Keras/TF sang TF.js
    model = await tf.loadLayersModel(MODEL_PATH);
    console.log("Mô hình đã tải thành công.");
}

// --- 2. XỬ LÝ ẢNH & DỰ ĐOÁN ---

fileUpload.addEventListener('change', (event) => {
    if (event.target.files.length === 0) return;
    
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        imgElement.src = e.target.result;
        imgElement.style.display = 'block';
        placeholder.style.display = 'none';
        
        stopCamera(); 
        predict(imgElement);
    };
    reader.readAsDataURL(file);
});


function preprocess(img) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(img);

        // Kích thước Model (Cần kiểm tra lại kích thước chính xác của model bạn)
        const resized = tf.image.resizeBilinear(tensor, [256, 256]);
        
        // Chuẩn hóa (0-1)
        const normalized = resized.toFloat().div(tf.scalar(255));
        
        // Thêm chiều batch (1, 256, 256, 3)
        const batched = normalized.expandDims(0);
        
        return batched;
    });
}

async function predict(img) {
    if (!model) {
        alert("Mô hình AI chưa sẵn sàng. Vui lòng chờ.");
        return;
    }

    loadingBar.classList.remove('hidden');
    predClassSpan.innerText = "Đang xử lý...";
    confidenceSpan.innerText = "---";
    progressBar.set(0); 

    try {
        const preprocessedImage = preprocess(img);
        
        const prediction = await model.predict(preprocessedImage).data();
        const maxPrediction = Math.max(...prediction);
        const predictionIndex = prediction.indexOf(maxPrediction);
        
        // Lấy tên bệnh và ID
        const idString = String(predictionIndex);
        const diseaseName = class_indices[idString] || "Không xác định";

        // Cập nhật kết quả
        if (maxPrediction >= THRESHOLD && disease_protocols_map[idString]) {
            predClassSpan.innerText = diseaseName;
            confidenceSpan.innerText = `${Math.round(maxPrediction * 100)}%`;
            lastPredictionId = idString;
            
        } else {
            predClassSpan.innerText = "Không rõ bệnh";
            confidenceSpan.innerText = `${Math.round(maxPrediction * 100)}%`;
            lastPredictionId = null;
        }
        
        // Cập nhật progress bar
        progressBar.animate(maxPrediction);

    } catch (error) {
        console.error("Lỗi dự đoán:", error);
        predClassSpan.innerText = "Lỗi dự đoán";
        confidenceSpan.innerText = "---";
    } finally {
        loadingBar.classList.add('hidden');
    }
}

// --- 3. XỬ LÝ CAMERA (Giữ nguyên logic của bạn) ---

async function startCamera() {
    // ... (Giữ nguyên logic startCamera) ...
    try {
        if (currentStream) currentStream.getTracks().forEach(track => track.stop());

        currentStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });

        videoStream.srcObject = currentStream;
        cameraContainer.classList.remove('hidden');
        placeholder.style.display = 'none';
        imgElement.style.display = 'none';
        cameraStatus.innerText = "Camera đang hoạt động...";
        captureButton.removeAttribute('disabled');
        
        // Lệnh này được thêm vào để đảm bảo video chạy trước khi chụp
        videoStream.play();

    } catch (err) {
        console.error("Lỗi truy cập Camera:", err);
        alert("Không thể truy cập camera. Vui lòng kiểm tra quyền truy cập.");
        stopCamera();
    }
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    
    cameraContainer.classList.add('hidden');
    placeholder.style.display = 'block';
    captureButton.setAttribute('disabled', 'true');
    cameraStatus.innerText = "";
}

// Event Listeners Camera
cameraToggle.addEventListener('click', () => {
    if (cameraContainer.classList.contains('hidden')) {
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
    imgElement.src = canvas.toDataURL('image/jpeg', 0.9); 
    imgElement.style.display = 'block';
    placeholder.style.display = 'none';
    
    stopCamera(); // Tắt camera sau khi chụp
    
    predict(imgElement);
});

// --- 4. HIỂN THỊ CHI TIẾT VÀ PHẠM VI ---

document.getElementById('detailBtn').addEventListener('click', showDetail);

function showDetail() {
    if (lastPredictionId && disease_protocols_map[lastPredictionId]) {
        renderProtocolDetail(disease_protocols_map[lastPredictionId]);
        detailOverlay.classList.remove('hidden');
    } else {
        alert("Không có kết quả dự đoán để hiển thị chi tiết.");
    }
}

closeDetailBtn.addEventListener('click', () => {
    detailOverlay.classList.add('hidden');
});

// Hàm hiển thị chi tiết (Giữ nguyên logic cũ, chỉ thay đổi tên thuộc tính)
function renderProtocolDetail(protocol) {
    const tacNhan = protocol.I_Tac_Nhan_Chu_Ky_Dieu_Kien;
    const canhTac = protocol.II_Bien_Phap_Canh_Tac_Chuyen_Sau;
    const hoaHoc = protocol.III_Chien_Luoc_Kiem_Soat_Hoa_Hoc;
    const nguonTK = protocol.IV_Nguon_Tham_Khao_Uy_Tin;

    detailContent.innerHTML = `
        <div class="protocol-header">
            <h3>${protocol.Ten_Benh_Tieng_Viet}</h3>
            <p><strong>Tên Khoa Học:</strong> ${protocol.Ten_Khoa_Hoc}</p>
            <p><strong>Nhóm Cây:</strong> ${protocol.Phan_Loai_Nhom_Cay}</p>
        </div>

        <details class="detail-section" open>
            <summary>I. Tác Nhân & Điều Kiện Gây Bệnh</summary>
            <div class="detail-content">
                <p>• <strong>Tác nhân:</strong> ${tacNhan.Tac_Nhan_Sinh_Hoc}</p>
                <p>• <strong>Cơ chế lây lan:</strong> ${tacNhan.Co_Che_Lay_Lan}</p>
                <p>• <strong>Điều kiện tối ưu:</strong> ${tacNhan.Nhiet_Do_Thoi_DIem_Toi_Uu}</p>
                <p>• <strong>Chẩn đoán chuyên sâu:</strong> ${tacNhan.Dau_Hieu_Chuan_Doan_Chuyen_Sau}</p>
            </div>
        </details>
        
        <details class="detail-section">
            <summary>II. Biện Pháp Canh Tác Tổng Hợp</summary>
            <div class="detail-content">
                <p>• <strong>Giống:</strong> ${canhTac.Giong_Khang_Benh}</p>
                <p>• <strong>Quản lý tàn dư:</strong> ${canhTac.Quan_Ly_Tan_Du_Dat}</p>
                <p>• <strong>Quản lý nước:</strong> ${canhTac.Quan_Ly_Nuoc_Tuoi}</p>
                <p>• <strong>Dinh dưỡng:</strong> ${canhTac.Quan_Ly_Dinh_Duong_Tong_Hop}</p>
            </div>
        </details>

        <details class="detail-section">
            <summary>III. Chiến Lược Kiểm Soát Hóa Học</summary>
            <div class="detail-content">
                <p>• <strong>Phòng ngừa:</strong> <span class="highlight-chem">${hoaHoc.Hoat_Chat_Phong_Ngua}</span></p>
                <p>• <strong>Điều trị:</strong> <span class="highlight-chem">${hoaHoc.Hoat_Chat_Dieu_Tri_Tru_Khuan}</span></p>
                <p class="warning-text"><i class="material-icons">warning</i>${hoaHoc.Luu_Y_Tong_Hop}</p>
            </div>
        </details>

        <details class="detail-section">
            <summary>IV. Nguồn Tham Khảo</summary>
            <div class="detail-content">
                <ul class="source-list">
                    ${nguonTK.map(src => `
                        <li><a href="${src.URL}" target="_blank"><i class="material-icons">link</i> ${src.Ten_Nguon}</a></li>
                    `).join('')}
                </ul>
            </div>
        </details>
    `;
}

// Xử lý Phạm vi (Scope List)
scopeBtn.addEventListener('click', () => {
    renderScopeList();
    scopeOverlay.classList.remove('hidden');
});

closeScopeBtn.addEventListener('click', () => {
    scopeOverlay.classList.add('hidden');
});

function renderScopeList() {
    scopeContent.innerHTML = '';
    
    // Sắp xếp theo ID số để hiển thị thứ tự chuẩn (0, 1, 2...)
    const sortedKeys = Object.keys(disease_protocols_map).sort((a, b) => parseInt(a) - parseInt(b));

    sortedKeys.forEach(key => {
        const item = disease_protocols_map[key];
        const div = document.createElement('div');
        div.className = 'scope-item';
        
        // Khắc phục lỗi undefined.png bằng cách dùng đúng Ten_Benh_Tieng_Viet
        const name = item.Ten_Benh_Tieng_Viet || `Bệnh ID ${key}`; 

        // Đường dẫn ảnh (Giả định ảnh có tên 0.png, 1.jpg,...)
        const pngPath = `./images/${key}.png`;
        const jpgPath = `./images/${key}.jpg`;

        div.innerHTML = `
            <div class="scope-img-wrapper">
                <img src="${pngPath}" 
                     onerror="this.onerror=null; this.src='${jpgPath}';" 
                     alt="${name}"
                     loading="lazy">
            </div>
            <div class="scope-name">${name}</div>
        `;
        
        div.addEventListener('click', () => {
            renderProtocolDetail(item);
            detailOverlay.classList.remove('hidden');
            scopeOverlay.classList.add('hidden'); 
        });

        scopeContent.appendChild(div);
    });
}

// --- CHẾ ĐỘ TỐI / SÁNG ---
modeToggle.addEventListener('click', () => {
    if (body.classList.contains('light-mode')) {
        body.classList.replace('light-mode', 'dark-mode');
        modeToggle.innerHTML = '<i class="material-icons">wb_sunny</i> Chế độ Sáng';
    } else {
        body.classList.replace('dark-mode', 'light-mode');
        modeToggle.innerHTML = '<i class="material-icons">brightness_4</i> Chế độ Tối';
    }
});