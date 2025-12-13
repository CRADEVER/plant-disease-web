/* =========================================
   IPDM SYSTEM - LOGIC CORE (Phiên bản Ảnh Phạm vi - Đã sửa lỗi Key JSON)
   ========================================= */

// --- Biến toàn cục ---
let model;
let disease_protocols_map = {}; 
let class_indices = {}; 
let currentStream = null;
let isScanning = false;
let scanInterval = null;
let lastPredictionId = null; 

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

// --- 1. KHỞI TẠO & EFFECT ---

$(document).ready(function() {
    // Kích hoạt hiệu ứng mặt nước trên background (nen.png)
    try {
        $('#ripple-background').ripples({
            resolution: 512,
            dropRadius: 20,
            perturbance: 0.04,
        });
    } catch (e) {
        console.log("Ripples effect error:", e);
    }
});

async function initialize() {
    try {
        await fetchData();
        await loadModel();
    } catch (e) {
        console.error(e);
        systemStatus.innerText = "Lỗi khởi động hệ thống.";
        systemStatus.style.background = "#f44336";
    }
}

async function fetchData() {
    try {
        let response = await fetch('./class_indices.json');
        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
        let data = await response.json();
        
        const protocolsArray = data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet;
        if (Array.isArray(protocolsArray)) {
            protocolsArray.forEach(item => {
                // SỬA LỖI Ở ĐÂY: Dùng đúng key trong JSON (Ma_ID thay vì Mã_ID)
                disease_protocols_map[item.Ma_ID] = item;
                class_indices[item.Ma_ID] = item.Ten_Benh_Tieng_Viet; 
            });
            console.log("Dữ liệu đã tải: ", Object.keys(disease_protocols_map).length);
        }
    } catch (error) {
        console.error("Data Error:", error);
        throw error;
    }
}

async function loadModel() {
    try {
        model = await tf.loadGraphModel('./tensorflowjs-model/model.json'); 
        // Warm-up
        const dummy = tf.zeros([1, 224, 224, 3]);
        model.predict(dummy).dispose();
        dummy.dispose();

        systemStatus.innerText = "Sẵn sàng | Chọn phương thức";
        systemStatus.style.background = "#4CAF50";
    } catch (error) {
        console.error("Model Error:", error);
        systemStatus.innerText = "Lỗi tải Model AI";
        systemStatus.style.background = "#f44336";
        throw error;
    }
}

// --- 2. XỬ LÝ DỰ ĐOÁN (CORE AI) ---

async function predict(sourceElement) {
    if (!model) return;

    try {
        const tensor = tf.browser.fromPixels(sourceElement)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .div(tf.scalar(255.0))
            .expandDims();

        const predictions = await model.predict(tensor).data();
        const maxPrediction = Math.max(...predictions);
        const maxIndex = predictions.indexOf(maxPrediction);
        
        tensor.dispose();

        // Ngưỡng tin cậy (Ví dụ: > 45%)
        if (maxPrediction > 0.45) {
            const idString = String(maxIndex);
            const diseaseName = class_indices[idString] || "Không xác định";
            
            resultText.innerText = diseaseName;
            resultBtn.classList.remove('hidden');
            lastPredictionId = idString; 
            
            // Nếu độ tin cậy rất cao, đổi màu status
            systemStatus.innerText = `Phát hiện: ${diseaseName} (${Math.round(maxPrediction*100)}%)`;
        } else {
            resultText.innerText = "Chưa rõ bệnh...";
            lastPredictionId = null;
            systemStatus.innerText = "Đang quét...";
        }

    } catch (error) {
        console.error("Predict Error", error);
    }
}

// --- 3. CAMERA LOGIC (QUÉT) ---

async function startCameraScanning() {
    stopCamera(); 
    displayImage.classList.add('hidden');
    placeholderUI.classList.add('hidden');
    videoStream.classList.remove('hidden');
    resultBtn.classList.add('hidden'); 

    try {
        const constraints = { 
            video: { 
                facingMode: 'environment', 
                width: { ideal: 640 },
                height: { ideal: 480 }
            } 
        };
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        videoStream.srcObject = currentStream;
        
        videoStream.onloadedmetadata = () => {
            isScanning = true;
            systemStatus.innerText = "Đang quét... Giữ chắc tay";
            
            // Quét mỗi 500ms
            scanInterval = setInterval(() => {
                if (isScanning && model) {
                    predict(videoStream);
                }
            }, 500); 
        };

    } catch (err) {
        console.error(err);
        systemStatus.innerText = "Lỗi Camera";
        alert("Không thể truy cập Camera. Vui lòng kiểm tra quyền.");
    }
}

function stopCamera() {
    isScanning = false;
    if (scanInterval) clearInterval(scanInterval);
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    videoStream.classList.add('hidden');
}

cameraToggle.addEventListener('click', () => {
    if (isScanning) {
        stopCamera();
        placeholderUI.classList.remove('hidden');
        systemStatus.innerText = "Đã dừng quét";
        resultBtn.classList.add('hidden');
    } else {
        startCameraScanning();
    }
});

// --- 4. UPLOAD LOGIC ---

uploadInput.addEventListener('change', function (e) {
    const file = this.files[0];
    if (file) {
        stopCamera(); 
        
        const reader = new FileReader();
        reader.onload = function (evt) {
            displayImage.src = evt.target.result;
            displayImage.classList.remove('hidden');
            videoStream.classList.add('hidden');
            placeholderUI.classList.add('hidden');
            resultBtn.classList.add('hidden');
            
            systemStatus.innerText = "Đang xử lý ảnh...";

            // Chờ ảnh load xong mới predict
            displayImage.onload = () => {
                predict(displayImage);
                systemStatus.innerText = "Hoàn tất phân tích";
            };
        };
        reader.readAsDataURL(file);
    }
});

// --- 5. OVERLAY LOGIC ---

resultBtn.addEventListener('click', () => {
    if (lastPredictionId && disease_protocols_map[lastPredictionId]) {
        renderProtocolDetail(disease_protocols_map[lastPredictionId]);
        detailOverlay.classList.remove('hidden');
    }
});

closeDetailBtn.addEventListener('click', () => {
    detailOverlay.classList.add('hidden');
});

// Hàm hiển thị chi tiết phác đồ
function renderProtocolDetail(protocol) {
    // SỬA LỖI KEY Ở ĐÂY: Dùng key không dấu (Ten_Benh_Tieng_Viet)
    const diseaseName = protocol.Ten_Benh_Tieng_Viet;
    const isHealthy = diseaseName.toLowerCase().includes("khỏe mạnh");
    
    // Cập nhật các key truy cập object con
    const tacNhan = protocol.I_Tac_Nhan_Chu_Ky_Dieu_Kien;
    const canhTac = protocol.II_Bien_Phap_Canh_Tac_Chuyen_Sau;
    const hoaHoc = protocol.III_Chien_Luoc_Kiem_Soat_Hoa_Hoc;

    let html = `
        <div class="protocol-header">
            <h3>${diseaseName}</h3>
            <span class="status-pill" style="background:${isHealthy ? '#4CAF50' : '#FF9800'}">
                ${protocol.Phan_Loai_Nhom_Cay}
            </span>
        </div>
        <div class="detail-body">
            <details class="protocol-detail-section" open>
                <summary><i class="material-icons">search</i> Chẩn đoán & Tác nhân</summary>
                <div class="detail-content">
                    <p><strong>Tác nhân:</strong> ${tacNhan?.Tac_Nhan_Sinh_Hoc || 'N/A'}</p>
                     <div class="symptom-box">
                        <strong>Dấu hiệu:</strong><br>
                        ${tacNhan?.Dau_Hieu_Chuan_Doan_Chuyen_Sau ? tacNhan.Dau_Hieu_Chuan_Doan_Chuyen_Sau.replace(/\n/g, '<br>') : 'N/A'}
                    </div>
                </div>
            </details>
    `;

    if (canhTac) {
        html += `
            <details class="protocol-detail-section">
                <summary><i class="material-icons">agriculture</i> Biện pháp Canh tác</summary>
                <div class="detail-content">
                    <p>• <strong>Vệ sinh:</strong> ${canhTac.Quan_Ly_Tan_Du_Dat || 'N/A'}</p>
                    <p>• <strong>Tưới tiêu:</strong> ${canhTac.Quan_Ly_Nuoc_Tuoi || 'N/A'}</p>
                    <p>• <strong>Dinh dưỡng:</strong> ${canhTac.Quan_Ly_Dinh_Duong_Tong_Hop || 'N/A'}</p>
                </div>
            </details>
        `;
    }

    if (!isHealthy && hoaHoc) {
        html += `
            <details class="protocol-detail-section">
                <summary><i class="material-icons">science</i> Thuốc BVTV (Hóa học)</summary>
                <div class="detail-content">
                    <p style="color:#d32f2f"><strong>Phòng ngừa:</strong> ${hoaHoc.Hoat_Chat_Phong_Ngua || 'N/A'}</p>
                    <p style="color:#d32f2f"><strong>Điều trị:</strong> ${hoaHoc.Hoat_Chat_Dieu_Tri_Tru_Khuan || 'N/A'}</p>
                    <p><em>Lưu ý: ${hoaHoc.Luu_Y_Tong_Hop}</em></p>
                </div>
            </details>
        `;
    }

    html += `</div>`;
    detailContent.innerHTML = html;
}

// 5b. Hiển thị Phạm vi (Scope) - UPDATED FOR IMAGES
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
        
        // Đường dẫn ảnh (sử dụng key là ID, ví dụ 0.png, 1.png)
        const pngPath = `./images/${key}.png`;
        const jpgPath = `./images/${key}.jpg`;

        // SỬA LỖI KEY: item.Ten_Benh_Tieng_Viet
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
        });

        scopeContent.appendChild(div);
    });
}

// --- Init ---
document.addEventListener('DOMContentLoaded', initialize);