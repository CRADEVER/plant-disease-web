let model;
let class_indices; // Map: Mã_ID (string) -> Tên_Bệnh
let disease_data; // Raw JSON Data for details lookup
let model_init_promise; // Promise để theo dõi trạng thái tải model (Tải 1 lần)

// Các biến DOM cần thiết cho UI
let fileUpload = document.getElementById('uploadImage');
let img = document.getElementById('image');
let boxResult = document.querySelector('.box-result');
let confidence = document.querySelector('.confidence');
let pconf = document.querySelector('.box-result p');
let modeToggle = document.getElementById('modeToggle');
let body = document.body;

// Biến cho Camera (từ file index.html)
let cameraToggle = document.getElementById('cameraToggle');
let cameraContainer = document.getElementById('cameraContainer');
let videoStream = document.getElementById('videoStream');
let captureButton = document.getElementById('captureButton');
let stopButton = document.getElementById('stopButton');
let cameraStatus = document.getElementById('cameraStatus');
let canvas = document.getElementById('canvas');
let context = canvas.getContext('2d');
let currentStream;

// Biến cho Kết quả và Chi tiết
let resultContainer = document.querySelector('.box-result'); // Sử dụng lại boxResult
let diseaseDetails = document.getElementById('diseaseDetails'); // Nếu bạn có phần tử này
let detailContent = document.getElementById('detailContent'); // Nếu bạn có phần tử này

let progressBar =
    new ProgressBar.Circle('#progress', {
    color: 'limegreen',
    strokeWidth: 10,
    duration: 2000,
    easing: 'easeInOut'
});

// --- LỚP DỮ LIỆU ---

async function fetchData(){
    // Dùng file JSON chi tiết tiếng Việt
    let response = await fetch('./class_indices.json');
    let data = await response.json();
    
    // Lưu trữ dữ liệu chi tiết
    let rawData = data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet; 
   
    let indices = {};
    // Tạo map: Mã_ID (string) -> Tên_Bệnh để lookup nhanh
    for (const item of rawData) {
        indices[item.Mã_ID] = item.Tên_Bệnh; 
    }
    return {indices: indices, rawData: rawData};
}


async function initialize() {
    let status = document.querySelector('.init_status');
    status.innerHTML = 'Đang tải Mô hình & Dữ liệu... <span class="fa fa-spinner fa-spin"></span>';
    if (boxResult) boxResult.style.display = 'block';

    try {
        // Tải Model và Data song song (chỉ 1 lần)
        const [modelLoad, dataLoad] = await Promise.all([
            tf.loadLayersModel('./tensorflowjs-model/model.json'), 
            fetchData() 
        ]);
        
        model = modelLoad;
        class_indices = dataLoad.indices;
        disease_data = dataLoad.rawData;
        
        status.innerHTML = 'Hệ thống đã sẵn sàng! Tải ảnh lên hoặc dùng Camera. <span class="fa fa-check"></span>';
        console.log('Model và dữ liệu đã tải xong.');
    } catch (error) {
        status.innerHTML = `Lỗi khởi tạo: ${error.message}. Vui lòng kiểm tra đường dẫn model và file json.`;
        console.error('Initialization error:', error);
    }
}

// Hàm format nội dung chi tiết từ JSON thành HTML
function formatDetailsToHtml(diseaseItem) {
    let html = '';
    
    function traverse(obj, title) {
        let content = '';

        if (title) {
            content += `<h3 class="detail-section-title">${title.replace(/_/g, ' ')}</h3>`;
        }

        if (Array.isArray(obj)) {
            // Xử lý danh sách Phác đồ Giai đoạn Cây
            for (const item of obj) {
                content += `<div class="phase-block">`;
                content += `<p><strong>Giai đoạn:</strong> <span>${item.Giai_đoạn || 'N/A'}</span></p>`;
                content += `<p><strong>Hoạt chất:</strong> <span>${item.Hoạt_chất_Đề_xuất || 'N/A'}</span></p>`;
                content += `<p><strong>Nhóm FRAC/IRAC:</strong> <span>${item.Nhóm_FRAC_IRAC || 'N/A'}</span></p>`;
                content += `<p><strong>Lưu ý:</strong> <em>${item.Lưu_ý_Ứng_dụng || 'Không'}</em></p>`;
                content += `</div>`;
            }
        } else if (typeof obj === 'object' && obj !== null) {
            for (const key in obj) {
                if (obj.hasOwnProperty(key)) {
                    let value = obj[key];
                    let formattedKey = key.replace(/_/g, ' ');

                    if (typeof value === 'object' && value !== null) {
                        content += traverse(value, key);
                    } else {
                        if (formattedKey === 'Phân loại' || formattedKey === 'Tên Bệnh' || formattedKey === 'Mã ID') {
                            // Bỏ qua các field đã hiển thị ở trên cùng
                        } else {
                            content += `<p><strong>${formattedKey}:</strong> <span>${value}</span></p>`;
                        }
                    }
                }
            }
        }
        
        // Chỉ thêm div bao bọc nếu có tiêu đề và không phải là list
        if (title && !Array.isArray(obj)) {
            html += `<div class="detail-section">${content}</div>`;
        } else {
            html += content;
        }
    }
    
    // Header chính
    html += `<h2>Bệnh: ${diseaseItem.Tên_Bệnh}</h2>`;
    html += `<p><strong>Phân loại:</strong> <span>${diseaseItem.Phân_loại}</span></p>`;
    html += `<hr>`;
    
    // Duyệt qua các phần chi tiết khác
    const keysToExclude = ['Tên_Bệnh', 'Mã_ID', 'Phân_loại'];
    for (const key in diseaseItem) {
        if (diseaseItem.hasOwnProperty(key) && !keysToExclude.includes(key)) {
            traverse(diseaseItem[key], key);
        }
    }
    
    return html;
}

function displayResult(resultID, confidence_val) {
    // Hiển thị container kết quả
    if (resultContainer) resultContainer.style.display = 'block'; 

    const diseaseName = class_indices[resultID];
    // Tìm chi tiết bệnh trong mảng raw data
    const diseaseItem = disease_data.find(item => item.Mã_ID == resultID);

    document.querySelector('.pred_class').innerHTML = diseaseName || 'Không xác định';
    confidence.innerHTML = confidence_val.toFixed(2);
    document.querySelector('.inner').innerHTML = `${confidence_val.toFixed(2)}%`;
    if (pconf) pconf.style.display = 'block';
    
    progressBar.set(0); 
    progressBar.animate(confidence_val / 100);

    // Hiển thị nội dung phân tích chi tiết trọn vẹn
    if (diseaseItem && detailContent) {
        detailContent.innerHTML = formatDetailsToHtml(diseaseItem);
        // Cần có phần tử có id="diseaseDetails" trong index.html
        // Nếu không có, bạn có thể chỉ hiển thị boxResult và đặt chi tiết vào đó
    } else if (detailContent) {
        detailContent.innerHTML = '<p>Không tìm thấy phác đồ quản lý chi tiết cho bệnh này.</p>';
    }
    
    document.querySelector('.init_status').innerHTML = '';
}


// --- LỚP DỰ ĐOÁN ---

async function predict() {
    // 1. Chắc chắn model đã tải xong
    if (!model) {
        let status = document.querySelector('.init_status');
        status.innerHTML = 'Hệ thống đang tải model, vui lòng đợi... <span class="fa fa-spinner fa-spin"></span>';
        await model_init_promise; 
    }
    
    // 2. Tiền xử lý ảnh: Kích thước 224x224, KHÔNG chuẩn hóa (LỖI ĐÃ KHẮC PHỤC)
    let tensorImg = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224]) 
        .toFloat() // Giá trị [0, 255]
        .expandDims(); // Shape (1, 224, 224, 3)
    
    // 3. Dự đoán
    let predictions = await model.predict(tensorImg).data();

    // 4. Hậu xử lý
    let predicted_index = tf.argMax(predictions).dataSync()[0];
    let confidence_value = predictions[predicted_index] * 100;
    
    // 5. Hiển thị kết quả (chuyển index thành string để khớp với Mã_ID trong JSON)
    displayResult(predicted_index.toString(), confidence_value);

    // Kích hoạt lại nút chụp nếu cần
    if (currentStream) {
        captureButton.disabled = false;
        cameraStatus.textContent = 'Camera đã sẵn sàng. Hãy chụp ảnh khác.';
    }
}


// ----------------------------------------------------------------
// EVENT LISTENERS & CAMERA LOGIC
// ----------------------------------------------------------------

fileUpload.addEventListener('change', function(e){

    // Dừng camera nếu đang chạy
    stopCamera();
    if (cameraContainer) cameraContainer.style.display = 'none';

    let file = this.files[0];
    if (file){
        // Cập nhật tên file (UI feedback)
        if (e.target.value) {
            document.getElementById("blankFile-1").innerHTML = e.target.value.replace("C:\\fakepath\\","");
            document.getElementById("choose-text-1").innerText = "Change Selected Image";
            document.querySelector(".success-1").style.display = "inline-block";
        }

        if (boxResult) boxResult.style.display = 'block';
        
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.addEventListener("load", function(){
            img.style.display = "block";
            img.setAttribute('src', this.result);
            img.style.width = "100%";
            img.style.height = "350px"; 
            
            // CHỈ GỌI PREDICT (FIX LỖI TẢI MODEL LẶP)
            predict(); 
        });
    }

    else{
        img.setAttribute("src", "");
        img.style.display = "none";
    }
})


// Logic cho Camera (Đảm bảo các nút và container có ID/class đúng trong index.html)
if (cameraToggle) {
    cameraToggle.addEventListener('click', function() {
        if (currentStream) {
            stopCamera();
            cameraContainer.style.display = 'none';
            cameraToggle.innerHTML = '<span class="camera-btn-text"><i class="material-icons d-block font-size-30">camera_alt</i> Mở Camera</span>';
        } else {
            cameraContainer.style.display = 'flex';
            cameraToggle.innerHTML = '<span class="camera-btn-text"><i class="material-icons d-block font-size-30">videocam_off</i> Đóng Camera</span>';
            startCamera();
        }
    });
}
if (stopButton) {
    stopButton.addEventListener('click', function() {
        stopCamera();
        cameraContainer.style.display = 'none';
        cameraToggle.innerHTML = '<span class="camera-btn-text"><i class="material-icons d-block font-size-30">camera_alt</i> Mở Camera</span>';
    });
}
if (captureButton) {
    captureButton.addEventListener('click', function() {
        if (currentStream) {
            captureButton.disabled = true;
            cameraStatus.textContent = 'Đang phân tích...';

            canvas.width = videoStream.videoWidth;
            canvas.height = videoStream.videoHeight;
            context.drawImage(videoStream, 0, 0, canvas.width, canvas.height);
            
            img.setAttribute('src', canvas.toDataURL('image/jpeg'));
            img.style.display = "block";
            img.style.width = "100%";
            img.style.height = "350px";

            stopCamera();
            cameraContainer.style.display = 'none';
            
            predict();
        }
    });
}

async function startCamera() {
    try {
        cameraStatus.textContent = 'Đang yêu cầu truy cập camera...';
        const constraints = { video: { facingMode: 'environment' } }; // Ưu tiên camera sau
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        videoStream.srcObject = currentStream;
        videoStream.play();
        cameraStatus.textContent = 'Camera đã sẵn sàng. Hãy chụp ảnh.';
        captureButton.disabled = false;
        videoStream.style.display = 'block';
        captureButton.style.display = 'block';
        stopButton.style.display = 'block';
        img.style.display = 'none'; 
        if (boxResult) boxResult.style.display = 'none'; 
    } catch (err) {
        cameraStatus.textContent = `Lỗi truy cập camera: ${err.name}.`;
        captureButton.disabled = true;
    }
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    if (videoStream) videoStream.srcObject = null;
    if (captureButton) captureButton.disabled = true;
    if (cameraStatus) cameraStatus.textContent = 'Camera đã dừng.';
}

// Logic chuyển đổi Dark/Light mode
if (modeToggle) {
    modeToggle.addEventListener('click', () => {
        if (body.classList.contains('light-mode')) {
            body.classList.replace('light-mode', 'dark-mode');
            modeToggle.innerHTML = '<i class="material-icons">wb_sunny</i> Chế độ Sáng';
        } else {
            body.classList.replace('dark-mode', 'light-mode');
            modeToggle.innerHTML = '<i class="material-icons">brightness_4</i> Chế độ Tối';
        }
    });
}


// --- CHẠY LẦN ĐẦU ---
// KHỞI TẠO MODEL 1 LẦN DUY NHẤT KHI TẢI TRANG
model_init_promise = initialize();
