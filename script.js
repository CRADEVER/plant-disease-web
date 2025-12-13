/* =========================================
   IPDM SYSTEM - LOGIC CORE (Phiên bản Ảnh Phạm vi)
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
                disease_protocols_map[item.Mã_ID] = item;
                class_indices[item.Mã_ID] = item.Tên_Bệnh; 
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
    const isHealthy = protocol.Tên_Bệnh.toLowerCase().includes("khỏe mạnh");
    
    let html = `
        <div class="protocol-header">
            <h3>${protocol.Tên_Bệnh}</h3>
            <span class="status-pill" style="background:${isHealthy ? '#4CAF50' : '#FF9800'}">
                ${protocol.Phân_loại || protocol.Phân_Loại_Nhom_Cay}
            </span>
        </div>
        <div class="detail-body">
            <details class="protocol-detail-section" open>
                <summary><i class="material-icons">search</i> Chẩn đoán & Tác nhân</summary>
                <div class="detail-content">
                    <p><strong>Tác nhân:</strong> ${protocol.I_Tác_Nhan_Chu_Ky_Dieu_Kien?.Tac_Nhan_Sinh_Hoc || 'N/A'}</p>
                     <div class="symptom-box">
                        <strong>Dấu hiệu:</strong><br>
                        ${protocol.I_Tác_Nhan_Chu_Ky_Dieu_Kien?.Dau_Hieu_Chuan_Doan_Chuyen_Sau ? protocol.I_Tác_Nhan_Chu_Ky_Dieu_Kien.Dấu_hiệu_Chẩn_đoán_Chuyên_sâu.replace(/\n/g, '<br>') : 'N/A'}
                    </div>
                </div>
            </details>
    `;

    if (protocol.II_Bien_Phap_Canh_Tac_Chuyen_Sau) {
        const cult = protocol.II_Bien_Phap_Canh_Tac_Chuyen_Sau;
        html += `
            <details class="protocol-detail-section">
                <summary><i class="material-icons">agriculture</i> Biện pháp Canh tác</summary>
                <div class="detail-content">
                    <p>• <strong>Vệ sinh:</strong> ${cult.Quan_Ly_Tan_Du_Dat || 'N/A'}</p>
                    <p>• <strong>Tưới tiêu:</strong> ${cult.Quan_Ly_Nuoc_Tuoi || 'N/A'}</p>
                    <p>• <strong>Dinh dưỡng:</strong> ${cult.Quan_Ly_Dinh_Duong_Tong_Hop || 'N/A'}</p>
                </div>
            </details>
        `;
    }

    if (!isHealthy && protocol.III_Chien_Luoc_Kiem_Soat_Hoa_Hoc) {
        const chem = protocol.III_Chien_Luoc_Kiem_Soat_Hoa_Hoc;
        html += `
            <details class="protocol-detail-section">
                <summary><i class="material-icons">science</i> Thuốc BVTV (Hóa học)</summary>
                <div class="detail-content">
                    <p style="color:#d32f2f"><strong>Phòng ngừa:</strong> ${chem.Hoat_Chat_Phong_Ngua || 'N/A'}</p>
                    <p style="color:#d32f2f"><strong>Điều trị:</strong> ${chem.Hoat_Chat_Dieu_Tri_Tru_Khuan || 'N/A'}</p>
                    <p><em>Lưu ý: ${chem.Luu_Y_Tong_Hop}</em></p>
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
        
        // Đường dẫn ảnh
        const pngPath = `./images/${key}.png`;
        const jpgPath = `./images/${key}.jpg`;

        // Logic HTML: Thử load PNG, lỗi thì load JPG (Fallback ngay trong thẻ img)
        div.innerHTML = `
            <div class="scope-img-wrapper">
                <img src="${pngPath}" 
                     onerror="this.onerror=null; this.src='${jpgPath}';" 
                     alt="${item.Tên_Bệnh}"
                     loading="lazy">
            </div>
            <div class="scope-name">${item.Tên_Bệnh}</div>
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