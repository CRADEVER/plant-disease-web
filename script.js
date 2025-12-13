/* =========================================
   IPDM SYSTEM - LOGIC CORE (Phiên bản Sửa lỗi)
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
const systemStatus = document.getElementById('systemStatus');
const displayImage = document.getElementById('displayImage');
const videoStream = document.getElementById('videoStream');
const placeholderUI = document.getElementById('placeholderUI');
const resultBtn = document.getElementById('resultBtn');
const resultText = document.getElementById('resultText');

// Buttons
const uploadInput = document.getElementById('uploadInput');
const cameraToggle = document.getElementById('cameraToggle');
const scopeBtn = document.getElementById('scopeBtn');

// Overlays
const detailOverlay = document.getElementById('detailOverlay');
const detailContent = document.getElementById('detailContent');
const closeDetailBtn = document.getElementById('closeDetailBtn');

const scopeOverlay = document.getElementById('scopeOverlay');
const scopeContent = document.getElementById('scopeContent');
const closeScopeBtn = document.getElementById('closeScopeBtn');

// --- Cần khởi tạo Canvas Context cho Camera Capture ---
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');


// --- 1. KHỞI TẠO & EFFECT ---

$(document).ready(function() {
    // Kích hoạt hiệu ứng mặt nước trên background (nen.png)
    try {
        // Tối ưu hóa tham số cho hiệu ứng rõ ràng hơn, không bị mờ
        $('#ripple-background').ripples({
            resolution: 512, 
            dropRadius: 20,  // Bán kính giọt nước (Tăng nhẹ)
            perturbance: 0.04, // Độ nhiễu/sóng lăn tăn (Giảm/tăng tùy ý để rõ hơn)
            interactive: true
        });
        console.log("Hiệu ứng mặt nước đã kích hoạt.");
    } catch (e) {
        console.error("Lỗi kích hoạt hiệu ứng Ripples:", e);
    }
    
    // Bắt đầu khởi tạo hệ thống
    initialize();
});

async function initialize() {
    systemStatus.innerText = "Đang tải dữ liệu bệnh...";
    try {
        await fetchData();
        systemStatus.innerText = "Đang tải mô hình AI...";
        await loadModel();
        
        systemStatus.innerText = "Hệ thống Sẵn sàng";
        systemStatus.style.backgroundColor = "#00a896"; // Màu xanh lá cây
        
        // Gắn listener cho Upload và Camera
        uploadInput.addEventListener('change', handleImageUpload);
        cameraToggle.addEventListener('click', toggleCamera);
        resultBtn.addEventListener('click', showDetail);

    } catch (e) {
        console.error("Lỗi khởi tạo hệ thống:", e);
        systemStatus.innerText = "Lỗi nghiêm trọng: Không thể khởi tạo hệ thống.";
        systemStatus.style.backgroundColor = "#f44336";
    }
}

// --- SỬA LỖI TẢI 65 MỤC DỮ LIỆU ---
async function fetchData() {
    try {
        let response = await fetch('./class_indices.json');
        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
        let data = await response.json();
        
        // Lấy mảng dữ liệu bệnh từ khóa chính (Đã xác minh trong file JSON)
        const protocolsArray = data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet; //
        
        if (Array.isArray(protocolsArray) && protocolsArray.length > 0) {
            protocolsArray.forEach(item => {
                // Sử dụng Ma_ID để làm key, đảm bảo nó là chuỗi
                const key = String(item.Ma_ID); 
                disease_protocols_map[key] = item;
                class_indices[key] = item.Ten_Benh_Tieng_Viet; 
            });
            console.log(`Dữ liệu đã tải thành công: ${Object.keys(disease_protocols_map).length} mục.`);
        } else {
             console.error("JSON Error: Mảng dữ liệu rỗng hoặc không phải mảng.");
             throw new Error("Dữ liệu JSON rỗng.");
        }
    } catch (error) {
        console.error("Data Error:", error);
        systemStatus.innerText = "Lỗi tải dữ liệu bệnh cây.";
        systemStatus.style.background = "#f44336";
        throw error;
    }
}

async function loadModel() {
    // Tải mô hình đã được chuyển đổi từ Keras/TF sang TF.js
    model = await tf.loadLayersModel(MODEL_PATH);
    console.log("Mô hình đã tải thành công.");
}

// --- 2. XỬ LÝ ẢNH & DỰ ĐOÁN ---

function handleImageUpload(event) {
    if (event.target.files.length === 0) return;
    
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        displayImage.src = e.target.result;
        displayImage.style.display = 'block';
        placeholderUI.style.display = 'none';
        videoStream.style.display = 'none';
        stopCamera(); // Đảm bảo tắt camera nếu đang mở
        predict(displayImage);
    };
    reader.readAsDataURL(file);
}

// Hàm xử lý ảnh trước khi đưa vào model (cần thay đổi nếu model yêu cầu kích thước/chuẩn hóa khác)
function preprocess(img) {
    return tf.tidy(() => {
        // Chuyển ảnh DOM sang Tensor
        let tensor = tf.browser.fromPixels(img);

        // Resize (Giả định model yêu cầu 256x256)
        const resized = tf.image.resizeBilinear(tensor, [256, 256]);
        
        // Chuẩn hóa (Giả định chuẩn hóa về 0-1)
        const normalized = resized.toFloat().div(tf.scalar(255));
        
        // Thêm chiều batch (1, 256, 256, 3)
        const batched = normalized.expandDims(0);
        
        return batched;
    });
}

async function predict(img) {
    if (!model) {
        alert("Mô hình AI chưa sẵn sàng. Vui lòng thử lại.");
        return;
    }

    // Hiển thị loading bar
    document.getElementById('loadingPredictionBar').classList.remove('hidden');
    resultText.innerText = "Đang xử lý...";
    resultBtn.classList.add('hidden');
    
    try {
        const preprocessedImage = preprocess(img);
        
        // Dự đoán
        const prediction = await model.predict(preprocessedImage).data();
        const maxPrediction = Math.max(...prediction);
        const predictionIndex = prediction.indexOf(maxPrediction);
        
        // Lấy tên bệnh và ID
        const idString = String(predictionIndex);
        const diseaseName = class_indices[idString] || "Không xác định";

        // Cập nhật kết quả
        if (maxPrediction >= THRESHOLD && disease_protocols_map[idString]) {
            resultText.innerHTML = `
                Bệnh: <b>${diseaseName}</b><br>
                Độ tin cậy: <b>${Math.round(maxPrediction * 100)}%</b>
            `;
            resultBtn.classList.remove('hidden');
            lastPredictionId = idString;
            systemStatus.innerText = `Phát hiện: ${diseaseName} (${Math.round(maxPrediction*100)}%)`;
            systemStatus.style.backgroundColor = "#00a896";
        } else {
            resultText.innerHTML = `
                Chưa rõ bệnh.<br>
                Độ tin cậy: <b>${Math.round(maxPrediction * 100)}%</b>
            `;
            lastPredictionId = null;
            systemStatus.innerText = "Đang chờ phát hiện...";
            systemStatus.style.backgroundColor = "#ffc107";
        }

    } catch (error) {
        console.error("Lỗi dự đoán:", error);
        resultText.innerText = "Lỗi trong quá trình dự đoán.";
    } finally {
        // Ẩn loading bar
        document.getElementById('loadingPredictionBar').classList.add('hidden');
    }
}

// --- 3. XỬ LÝ CAMERA ---

async function startCamera() {
    try {
        // Tắt camera cũ (nếu có)
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }

        currentStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment' // Ưu tiên camera sau trên điện thoại
            }
        });

        videoStream.srcObject = currentStream;
        videoStream.style.display = 'block';
        placeholderUI.style.display = 'none';
        displayImage.style.display = 'none';

        cameraToggle.innerHTML = '<i class="material-icons">videocam_off</i> Tắt Camera';

        // Bắt đầu quét tự động (Ví dụ: mỗi 5 giây)
        isScanning = true;
        scanInterval = setInterval(() => {
            if (videoStream.videoWidth > 0 && isScanning) {
                captureAndPredict();
            }
        }, 5000); // Tự động quét sau mỗi 5 giây

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
    
    clearInterval(scanInterval);
    isScanning = false;

    videoStream.style.display = 'none';
    placeholderUI.style.display = 'block';
    cameraToggle.innerHTML = '<i class="material-icons">qr_code_scanner</i> Quét Camera';
}

function toggleCamera() {
    if (!currentStream) {
        startCamera();
    } else {
        stopCamera();
    }
}

function captureAndPredict() {
    // Vẽ frame hiện tại lên canvas
    canvas.width = videoStream.videoWidth;
    canvas.height = videoStream.videoHeight;
    context.drawImage(videoStream, 0, 0, canvas.width, canvas.height);
    
    // Chuyển thành ảnh hiển thị (Tối ưu hóa chất lượng)
    displayImage.src = canvas.toDataURL('image/jpeg', 0.9); 
    displayImage.style.display = 'block';
    videoStream.style.display = 'none';
    
    // Dự đoán
    predict(displayImage);

    // Sau khi chụp 1 khung hình, quay lại hiển thị video stream
    videoStream.style.display = 'block';
    displayImage.style.display = 'none';
}


// --- 4. HIỂN THỊ PHÁC ĐỒ & PHẠM VI ---

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
                <p>• <strong>Điều kiện tối ưu:</strong> ${tacNhan.Nhiet_Do_Thoi_Diem_Toi_Uu}</p>
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
        
        // Đường dẫn ảnh (Giả định ảnh có tên 0.png, 1.jpg,...)
        const pngPath = `./images/${key}.png`;
        const jpgPath = `./images/${key}.jpg`;

        // Logic HTML: Thử load PNG, lỗi thì load JPG (Fallback ngay trong thẻ img)
        // **Lưu ý:** Tên thuộc tính trong JSON là Ten_Benh_Tieng_Viet, không phải Tên_Bệnh
        div.innerHTML = `
            <div class="scope-img-wrapper">
                <img src="${pngPath}" 
                     onerror="this.onerror=null; this.src='${jpgPath}';" 
                     alt="${item.Ten_Benh_Tieng_Viet}"
                     loading="lazy">
            </div>
            <div class="scope-name">${item.Ten_Benh_Tieng_Viet}</div>
        `;
        
        // Click vào item để xem chi tiết
        div.addEventListener('click', () => {
            renderProtocolDetail(item);
            detailOverlay.classList.remove('hidden');
            scopeOverlay.classList.add('hidden'); // Đóng Scope
        });

        scopeContent.appendChild(div);
    });
}