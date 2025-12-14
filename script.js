let model;
let disease_protocols_map = {}; 
let class_indices = {}; 
let currentStream = null;
let isScanning = false;
let scanInterval = null;
let lastPredictionId = null; 

const systemStatus = document.getElementById('systemStatus');
const displayImage = document.getElementById('displayImage');
const videoStream = document.getElementById('videoStream');
const placeholderUI = document.getElementById('placeholderUI');
const resultBtn = document.getElementById('resultBtn');
const resultText = document.getElementById('resultText');

const uploadInput = document.getElementById('uploadInput');
const cameraToggle = document.getElementById('cameraToggle');
const scopeBtn = document.getElementById('scopeBtn');

const detailOverlay = document.getElementById('detailOverlay');
const detailContent = document.getElementById('detailContent');
const closeDetailBtn = document.getElementById('closeDetailBtn');

const scopeOverlay = document.getElementById('scopeOverlay');
const scopeContent = document.getElementById('scopeContent');
const closeScopeBtn = document.getElementById('closeScopeBtn');

$(document).ready(function() {
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
                disease_protocols_map[item.Ma_ID] = item;
                class_indices[item.Ma_ID] = item.Ten_Benh_Tieng_Viet; 
            });
            console.log("Dữ liệu đã tải thành công:", Object.keys(disease_protocols_map).length, "bệnh.");
        }
    } catch (error) {
        console.error("Data Error:", error);
        throw error;
    }
}

async function loadModel() {
    try {
        model = await tf.loadGraphModel('./tensorflowjs-model/model.json'); 
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

        if (maxPrediction > 0.45) {
            const idString = String(maxIndex);
            const diseaseName = class_indices[idString] || "Không xác định";
            
            resultText.innerText = diseaseName;
            resultBtn.classList.remove('hidden');
            lastPredictionId = idString; 
            
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

            displayImage.onload = () => {
                predict(displayImage);
                systemStatus.innerText = "Hoàn tất phân tích";
            };
        };
        reader.readAsDataURL(file);
    }
});

resultBtn.addEventListener('click', () => {
    if (lastPredictionId && disease_protocols_map[lastPredictionId]) {
        renderProtocolDetail(disease_protocols_map[lastPredictionId]);
        detailOverlay.classList.remove('hidden');
    }
});

closeDetailBtn.addEventListener('click', () => {
    detailOverlay.classList.add('hidden');
});

function renderProtocolDetail(protocol) {
    const tenBenh = protocol.Ten_Benh_Tieng_Viet;
    const tenKhoaHoc = protocol.Ten_Khoa_Hoc;
    const phanLoai = protocol.Phan_Loai_Nhom_Cay;
    const isHealthy = tenBenh.toLowerCase().includes("khỏe mạnh");

    const muc1 = protocol.I_Tac_Nhan_Chu_Ky_Dieu_Kien || {};
    const muc2 = protocol.II_Bien_Phap_Canh_Tac_Chuyen_Sau || {};
    const muc3 = protocol.III_Chien_Luoc_Kiem_Soat_Hoa_Hoc || {};
    const muc4 = protocol.IV_Nguon_Tham_Khao_Uy_Tin || []; 

    let html = `
        <div class="protocol-header">
            <h3>${tenBenh}</h3>
            <p style="font-size: 0.9em; font-style: italic; color: #555;">${tenKhoaHoc}</p>
            <span class="status-pill" style="background:${isHealthy ? '#4CAF50' : '#FF9800'}">
                ${phanLoai}
            </span>
        </div>
        <div class="detail-body">
            
            <details class="protocol-detail-section" open>
                <summary><i class="material-icons">search</i> I. Tác nhân & Chu kỳ</summary>
                <div class="detail-content">
                    <p><strong>Tác nhân:</strong> ${muc1.Tac_Nhan_Sinh_Hoc || 'N/A'}</p>
                    <p><strong>Cơ chế lây lan:</strong> ${muc1.Co_Che_Lay_Lan || 'N/A'}</p>
                    <p><strong>Điều kiện:</strong> ${muc1.Nhiet_Do_Thoi_Diem_Toi_Uu || 'N/A'}</p>
                     <div class="symptom-box">
                        <strong>Dấu hiệu chuyên sâu:</strong><br>
                        ${muc1.Dau_Hieu_Chuan_Doan_Chuyen_Sau ? muc1.Dau_Hieu_Chuan_Doan_Chuyen_Sau.replace(/\n/g, '<br>') : 'N/A'}
                    </div>
                </div>
            </details>
    `;

    html += `
        <details class="protocol-detail-section">
            <summary><i class="material-icons">agriculture</i> II. Biện pháp Canh tác</summary>
            <div class="detail-content">
                <p>• <strong>Giống:</strong> ${muc2.Giong_Khang_Benh || 'N/A'}</p>
                <p>• <strong>Vệ sinh đất:</strong> ${muc2.Quan_Ly_Tan_Du_Dat || 'N/A'}</p>
                <p>• <strong>Tưới tiêu:</strong> ${muc2.Quan_Ly_Nuoc_Tuoi || 'N/A'}</p>
                <p>• <strong>Dinh dưỡng:</strong> ${muc2.Quan_Ly_Dinh_Duong_Tong_Hop || 'N/A'}</p>
            </div>
        </details>
    `;

    if (!isHealthy && (muc3.Hoat_Chat_Phong_Ngua || muc3.Hoat_Chat_Dieu_Tri_Tru_Khuan)) {
        html += `
            <details class="protocol-detail-section">
                <summary><i class="material-icons">science</i> III. Kiểm soát Hóa học</summary>
                <div class="detail-content">
                    <p style="color:#2E7D32"><strong>Hoạt chất Phòng ngừa:</strong> ${muc3.Hoat_Chat_Phong_Ngua || 'N/A'}</p>
                    <p style="color:#d32f2f"><strong>Hoạt chất Điều trị:</strong> ${muc3.Hoat_Chat_Dieu_Tri_Tru_Khuan || 'N/A'}</p>
                    <p><em>Lưu ý: ${muc3.Luu_Y_Tong_Hop || ''}</em></p>
                </div>
            </details>
        `;
    }
    
    if (Array.isArray(muc4) && muc4.length > 0) {
        let referenceList = '<ul class="reference-list">';
        muc4.forEach(ref => {
            if (ref.Ten_Nguon && ref.URL) {
                referenceList += `<li><a href="${ref.URL}" target="_blank">${ref.Ten_Nguon} <i class="material-icons" style="font-size:1em; vertical-align:middle;">link</i></a></li>`;
            }
        });
        referenceList += '</ul>';

        html += `
            <details class="protocol-detail-section">
                <summary><i class="material-icons">book</i> IV. Nguồn tham khảo</summary>
                <div class="detail-content">
                    ${referenceList}
                </div>
            </details>
        `;
    }


    html += `</div>`;
    detailContent.innerHTML = html;
}

scopeBtn.addEventListener('click', () => {
    renderScopeList();
    scopeOverlay.classList.remove('hidden');
});

closeScopeBtn.addEventListener('click', () => {
    scopeOverlay.classList.add('hidden');
});

function renderScopeList() {
    scopeContent.innerHTML = '';
    
    const sortedKeys = Object.keys(disease_protocols_map).sort((a, b) => parseInt(a) - parseInt(b));

    sortedKeys.forEach(key => {
        const item = disease_protocols_map[key];
        const div = document.createElement('div');
        div.className = 'scope-item';
        
        const paddedKey = String(key).padStart(4, '0');
        
        const jpgPath = `./images/${paddedKey}.jpg`;     
        const jpgUpperPath = `./images/${paddedKey}.JPG`; 

        div.innerHTML = `
            <div class="scope-img-wrapper">
                <img src="${jpgPath}" 
                     onerror="this.onerror=null; if(this.src.indexOf('.JPG') === -1) { this.src='${jpgUpperPath}'; }" 
                     alt="${item.Ten_Benh_Tieng_Viet}"
                     loading="lazy">
            </div>
            <div class="scope-name">${item.Ten_Benh_Tieng_Viet}</div>
        `;
        
        div.addEventListener('click', () => {
            renderProtocolDetail(item);
            detailOverlay.classList.remove('hidden'); 
        });

        scopeContent.appendChild(div);
    });
}

document.addEventListener('DOMContentLoaded', initialize);
