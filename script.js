let model;
let class_indices; // Map: Mã_ID (string) -> Tên_Bệnh
let disease_data; // Raw JSON Data for details lookup
let model_init_promise; // Promise để theo dõi trạng thái tải model

let fileUpload = document.getElementById('uploadImage');
let img = document.getElementById('image');
let boxResult = document.querySelector('.box-result');
let resultContainer = document.getElementById('resultContainer');
let diseaseDetails = document.getElementById('diseaseDetails');
let detailContent = document.getElementById('detailContent');

let confidence = document.querySelector('.confidence');
let pconf = document.querySelector('.box-result p');
let modeToggle = document.getElementById('modeToggle');
let body = document.body;


let cameraToggle = document.getElementById('cameraToggle');
let cameraContainer = document.getElementById('cameraContainer');
let videoStream = document.getElementById('videoStream');
let captureButton = document.getElementById('captureButton');
let stopButton = document.getElementById('stopButton');
let cameraStatus = document.getElementById('cameraStatus');
let canvas = document.getElementById('canvas');
let context = canvas.getContext('2d');
let currentStream;


let progressBar =
    new ProgressBar.Circle('#progress', {
    color: 'limegreen',
    strokeWidth: 10,
    duration: 2000,
    easing: 'easeInOut'
});

// Function để fetch dữ liệu JSON chi tiết
async function fetchData(){
    let response = await fetch('./class_indices.json');
    let data = await response.json();
    
    // Lưu trữ dữ liệu chi tiết
    disease_data = data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet; 
   
    let indices = {};
    for (const item of disease_data) {
        // Tạo map: Mã_ID (string) -> Tên_Bệnh
        indices[item.Mã_ID] = item.Tên_Bệnh; 
    }
    return {indices: indices, rawData: data};
}


async function initialize() {
    let status = document.querySelector('.init_status');
    status.innerHTML = 'Đang tải Mô hình & Dữ liệu... <span class="fa fa-spinner fa-spin"></span>';
    boxResult.style.display = 'block';

    try {
        // Tải Model và Data song song
        const [modelLoad, dataLoad] = await Promise.all([
            tf.loadLayersModel('./tensorflowjs-model/model.json'),
            fetchData() 
        ]);
        
        model = modelLoad;
        class_indices = dataLoad.indices;
        
        status.innerHTML = 'Hệ thống đã sẵn sàng! Tải ảnh lên hoặc dùng Camera. <span class="fa fa-check"></span>';
        console.log('Model và dữ liệu đã tải xong.');
    } catch (error) {
        status.innerHTML = `Lỗi khởi tạo: ${error.message}. Vui lòng kiểm tra đường dẫn model và file json.`;
        console.error('Initialization error:', error);
    }
}

// Hàm format nội dung chi tiết từ JSON thành HTML (Giữ nguyên logic đệ quy tốt)
function formatDetailsToHtml(diseaseItem) {
    let html = '';
    
    function traverse(obj, title) {
        let content = '';
        let isList = false;

        if (title) {
            content += `<h3 class="detail-section-title">${title.replace(/_/g, ' ')}</h3>`;
        }

        if (Array.isArray(obj)) {
            isList = true;
            for (const item of obj) {
                content += `<div class="phase-block">`;
                content += `<p><strong>Giai đoạn:</strong> ${item.Giai_đoạn || 'N/A'}</p>`;
                content += `<p><strong>Hoạt chất:</strong> ${item.Hoạt_chất_Đề_xuất || 'N/A'}</p>`;
                content += `<p><strong>Nhóm FRAC/IRAC:</strong> ${item.Nhóm_FRAC_IRAC || 'N/A'}</p>`;
                content += `<p><strong>Lưu ý:</strong> <em>${item.Lưu_ý_Ứng_dụng || 'Không'}</em></p>`;
                content += `</div>`;
            }
        } else if (typeof obj === 'object' && obj !== null) {
            for (const key in obj) {
                if (obj.hasOwnProperty(key)) {
                    let value = obj[key];
                    let formattedKey = key.replace(/_/g, ' ');

                    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                        content += traverse(value, key);
                    } else if (Array.isArray(value)) {
                        content += traverse(value, key);
                    } else {
                        if (formattedKey === 'Phân loại' || formattedKey === 'Tên Bệnh' || formattedKey === 'Mã ID') {
                            // Bỏ qua
                        } else {
                            content += `<p><strong>${formattedKey}:</strong> <span>${value}</span></p>`;
                        }
                    }
                }
            }
        } 
        
        if (title && !isList) {
            html += `<div class="detail-section">${content}</div>`;
        } else {
            html += content;
        }
    }
    
    html += `<h3>Bệnh: ${diseaseItem.Tên_Bệnh}</h3>`;
    html += `<p><strong>Phân loại:</strong> ${diseaseItem.Phân_loại}</p>`;
    
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
    boxResult.style.display = 'block';

    const diseaseName = class_indices[resultID];
    const diseaseItem = disease_data.find(item => item.Mã_ID == resultID);

    document.querySelector('.pred_class').innerHTML = diseaseName || 'Không xác định';
    confidence.innerHTML = confidence_val.toFixed(2);
    document.querySelector('.inner').innerHTML = `${confidence_val.toFixed(2)}%`;
    pconf.style.display = 'block';
    
    progressBar.set(0); 
    progressBar.animate(confidence_val / 100);

    // Hiển thị nội dung phân tích chi tiết trọn vẹn
    if (diseaseItem) {
        detailContent.innerHTML = formatDetailsToHtml(diseaseItem);
        if (diseaseDetails) diseaseDetails.style.display = 'block';
    } else {
        detailContent.innerHTML = '<p>Không tìm thấy phác đồ quản lý chi tiết cho bệnh này.</p>';
        if (diseaseDetails) diseaseDetails.style.display = 'block';
    }
    
    document.querySelector('.init_status').innerHTML = '';
}


async function predict() {
    // Bước 1: Chắc chắn model đã tải xong
    if (!model) {
        let status = document.querySelector('.init_status');
        status.innerHTML = 'Hệ thống đang tải model, vui lòng đợi... <span class="fa fa-spinner fa-spin"></span>';
        await model_init_promise; 
    }
    
    // Fix Preprocessing: Chuyển kích thước ảnh sang 256x256
    let tensorImg = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([256, 256]) // <--- ĐÃ SỬA TẠI ĐÂY
        .toFloat() // Chuyển sang Float32
        .div(tf.scalar(255.0)) // Chuẩn hóa về [0, 1]
        .expandDims(); 
    
    // Bước 2: Dự đoán
    let predictions = await model.predict(tensorImg).data();

    // Bước 3: Post-process
    let predicted_index = tf.argMax(predictions).dataSync()[0];
    let confidence_value = predictions[predicted_index] * 100;
    
    // Bước 4: Hiển thị kết quả (chuyển index thành string để khớp với Mã_ID trong JSON)
    displayResult(predicted_index.toString(), confidence_value);

    // Kích hoạt lại nút chụp nếu cần
    if (currentStream) {
        captureButton.disabled = false;
        cameraStatus.textContent = 'Camera đã sẵn sàng. Hãy chụp ảnh khác.';
    }
}

// ----------------------------------------------------------------
// EVENT LISTENERS 
// ----------------------------------------------------------------

fileUpload.addEventListener('change', function(e){

    stopCamera();
    cameraContainer.style.display = 'none';

    let file = this.files[0]
    if (file){
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.addEventListener("load", function(){
            img.style.display = "block"
            img.setAttribute('src', this.result);
            img.style.width = "100%";
            img.style.height = "350px"; 
            
            predict(); 
        });
    }

    else{
        img.setAttribute("src", "");
        img.style.display = "none";
    }
})


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

stopButton.addEventListener('click', function() {
    stopCamera();
    cameraContainer.style.display = 'none';
    cameraToggle.innerHTML = '<span class="camera-btn-text"><i class="material-icons d-block font-size-30">camera_alt</i> Mở Camera</span>';
});

captureButton.addEventListener('click', function() {
    if (currentStream) {
        captureButton.disabled = true;
        cameraStatus.textContent = 'Đang phân tích...';

        // Lấy khung hình từ video, không cần set kích thước 256x256 ở đây
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

// Khởi tạo model khi tải trang
model_init_promise = initialize();


async function startCamera() {
    try {
        cameraStatus.textContent = 'Đang yêu cầu truy cập camera...';
        
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'environment' 
            }
        };
        
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        videoStream.srcObject = currentStream;
        videoStream.play();
        cameraStatus.textContent = 'Camera đã sẵn sàng. Hãy chụp ảnh.';
        captureButton.disabled = false;
        videoStream.style.display = 'block';
        captureButton.style.display = 'block';
        stopButton.style.display = 'block';
        img.style.display = 'none'; 
        boxResult.style.display = 'none'; 
        if (resultContainer) resultContainer.style.display = 'none'; 
    } catch (err) {
        
        try {
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user' 
                }
            };
            currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            videoStream.srcObject = currentStream;
            videoStream.play();
            cameraStatus.textContent = 'Camera đã sẵn sàng. Hãy chụp ảnh.';
            captureButton.disabled = false;
            videoStream.style.display = 'block';
            captureButton.style.display = 'block';
            stopButton.style.display = 'block';
            img.style.display = 'none';
            boxResult.style.display = 'none';
            if (resultContainer) resultContainer.style.display = 'none';
        } catch (error) {
            cameraStatus.textContent = `Lỗi truy cập camera: ${error.name}. Vui lòng đảm bảo camera được phép sử dụng.`;
            captureButton.disabled = true;
            videoStream.style.display = 'none';
            captureButton.style.display = 'none';
            stopButton.style.display = 'none';
        }
    }
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    videoStream.srcObject = null;
    captureButton.disabled = true;
    cameraStatus.textContent = 'Camera đã dừng.';
}


modeToggle.addEventListener('click', () => {
    if (body.classList.contains('light-mode')) {
        body.classList.replace('light-mode', 'dark-mode');
        modeToggle.innerHTML = '<i class="material-icons">wb_sunny</i> Chế độ Sáng';
    } else {
        body.classList.replace('dark-mode', 'light-mode');
        modeToggle.innerHTML = '<i class="material-icons">brightness_4</i> Chế độ Tối';
    }
});
