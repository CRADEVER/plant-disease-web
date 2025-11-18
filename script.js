let model;
let class_indices; // Map: Index -> Disease Name
let disease_data; // Raw JSON Data for details lookup (Goal 3)
let fileUpload = document.getElementById('uploadImage');
let img = document.getElementById('image');
let boxResult = document.querySelector('.box-result'); // Prediction/Confidence box
let resultContainer = document.getElementById('resultContainer'); // New container for all results
let diseaseDetails = document.getElementById('diseaseDetails'); // New box for detailed analysis (Goal 5)
let detailContent = document.getElementById('detailContent'); // Container for the detail text
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

// Function để fetch dữ liệu JSON (class indices và chi tiết bệnh)
async function fetchData(){
    let response = await fetch('./class_indices.json');
    let data = await response.json();
    
    // Lưu trữ dữ liệu chi tiết
    disease_data = data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet; // Lấy mảng chi tiết
   
    let indices = {};
    for (const item of disease_data) {
        // Tạo map: Mã_ID -> Tên_Bệnh cho việc tra cứu kết quả predict
        indices[item.Mã_ID] = item.Tên_Bệnh; 
    }
    return indices; // Chỉ trả về map indices
}


async function initialize() {
    let status = document.querySelector('.init_status')
    status.innerHTML = 'Đang tải Mô hình & Dữ liệu... Vui lòng chờ.';
    boxResult.style.display = 'block';

    try {
        // ĐIỀU CHỈNH ĐƯỜNG DẪN TẠI ĐÂY: Thêm thư mục tensorflowjs-model/
        const [modelLoad, indicesLoad] = await Promise.all([
            tf.loadGraphModel('./tensorflowjs-model/model.json'), 
            fetchData() 
        ]);
        
        model = modelLoad;
        class_indices = indicesLoad;
        
        status.innerHTML = 'Hệ thống đã sẵn sàng! Tải ảnh lên hoặc dùng Camera.';
        console.log('Model và dữ liệu đã tải xong.');
    } catch (error) {
        status.innerHTML = `Lỗi khởi tạo: ${error.message}. Vui lòng kiểm tra file model và json.`;
        console.error('Initialization error:', error);
    }
}

// Hàm format nội dung chi tiết từ JSON thành HTML
function formatDetailsToHtml(diseaseItem) {
    let html = '';
    
    // Hàm đệ quy để duyệt qua các object lồng nhau
    function traverse(obj, title) {
        let content = '';
        let isList = false;

        // Dựa vào title của key cha để tạo tiêu đề
        if (title) {
            content += `<h3 class="detail-section-title">${title.replace(/_/g, ' ')}</h3>`;
        }

        // Trường hợp là một mảng (ví dụ: Phác đồ Giai đoạn Cây)
        if (Array.isArray(obj)) {
            isList = true;
            for (const item of obj) {
                // Tạo một khối riêng cho từng giai đoạn
                content += `<div class="phase-block">`;
                content += `<p><strong>Giai đoạn:</strong> ${item.Giai_đoạn || 'N/A'}</p>`;
                content += `<p><strong>Hoạt chất:</strong> ${item.Hoạt_chất_Đề_xuất || 'N/A'}</p>`;
                content += `<p><strong>Nhóm FRAC/IRAC:</strong> ${item.Nhóm_FRAC_IRAC || 'N/A'}</p>`;
                content += `<p><strong>Lưu ý:</strong> <em>${item.Lưu_ý_Ứng_dụng || 'Không'}</em></p>`;
                content += `</div>`;
            }
        } else if (typeof obj === 'object' && obj !== null) {
            // Trường hợp là object (ví dụ: Tác nhân, Biện pháp Canh tác)
            for (const key in obj) {
                if (obj.hasOwnProperty(key)) {
                    let value = obj[key];
                    let formattedKey = key.replace(/_/g, ' ');

                    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                        // Gọi đệ quy cho object con
                        content += traverse(value, key);
                    } else if (Array.isArray(value)) {
                         // Xử lý mảng (ví dụ: Phác đồ Giai đoạn Cây)
                        content += traverse(value, key);
                    } else {
                        // Trường hợp là cặp key-value thông thường
                        if (formattedKey === 'Phân loại') {
                            content += `<p><strong>Phân loại:</strong> <span>${value}</span></p>`;
                        } else if (formattedKey === 'Tên Bệnh' || formattedKey === 'Mã ID') {
                            // Bỏ qua vì đã hiển thị ở trên
                        } else {
                            content += `<p><strong>${formattedKey}:</strong> <span>${value}</span></p>`;
                        }
                    }
                }
            }
        } else {
            // Giá trị cuối cùng (không phải object hay array)
            content += `<p>${obj}</p>`;
        }
        
        // Bọc nội dung của object (trừ mảng/list)
        if (title && !isList) {
            html += `<div class="detail-section">${content}</div>`;
        } else {
            html += content;
        }
    }
    
    // Bắt đầu duyệt từ object gốc (bỏ qua Tên_Bệnh và Mã_ID)
    html += `<h3>Bệnh: ${diseaseItem.Tên_Bệnh}</h3>`;
    html += `<p><strong>Phân loại:</strong> ${diseaseItem.Phân_loại}</p>`;
    
    // Bỏ qua Tên_Bệnh, Mã_ID, Phân_loại đã xử lý
    const keysToExclude = ['Tên_Bệnh', 'Mã_ID', 'Phân_loại'];
    
    for (const key in diseaseItem) {
        if (diseaseItem.hasOwnProperty(key) && !keysToExclude.includes(key)) {
            traverse(diseaseItem[key], key);
        }
    }
    
    return html;
}

// Cập nhật hàm displayResult để hiển thị chi tiết
function displayResult(result, confidence_val) {
    const predClassElement = document.querySelector('.pred_class');
    const confidenceElement = document.querySelector('.confidence');
    
    // Hiển thị container kết quả
    resultContainer.style.display = 'block'; 
    boxResult.style.display = 'block';

    // 1. Lấy Tên bệnh từ index
    const diseaseName = class_indices[result];
    
    // 2. Tìm chi tiết bệnh trong mảng disease_data (đảm bảo phân tích chính xác)
    const diseaseItem = disease_data.find(item => item.Mã_ID == result);

    predClassElement.textContent = diseaseName || 'Không xác định';
    confidenceElement.textContent = confidence_val.toFixed(2);
    pconf.style.display = 'block';
    
    progressBar.animate(confidence_val / 100);

    // 3. Goal 3 & 5: Hiển thị nội dung phân tích chi tiết trọn vẹn
    if (diseaseItem) {
        detailContent.innerHTML = formatDetailsToHtml(diseaseItem);
        diseaseDetails.style.display = 'block';
    } else {
        detailContent.innerHTML = '<p>Không tìm thấy phác đồ quản lý chi tiết cho bệnh này.</p>';
        diseaseDetails.style.display = 'block'; // Vẫn hiển thị box, nhưng báo lỗi
    }
}


async function predict(imgElement) {
    boxResult.style.display = 'none'; // Ẩn kết quả cũ

    // 1. Preprocess
    let tensor = tf.browser.fromPixels(imgElement)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();
    
    // 2. Prediction
    let predictions = await model.predict(tensor).data();
    
    // 3. Post-process
    let predicted_index = tf.argMax(predictions).dataSync()[0];
    let confidence_value = predictions[predicted_index] * 100;

    // 4. Display
    displayResult(predicted_index.toString(), confidence_value);

    // Kích hoạt lại nút chụp sau khi phân tích xong
    if (currentStream) {
        captureButton.disabled = false;
        cameraStatus.textContent = 'Camera đã sẵn sàng. Hãy chụp ảnh khác.';
    }
}

// ... (Khai báo các event listeners) ...

fileUpload.addEventListener('change', (e) => {
    // ... (Code xử lý file upload, sau đó gọi predict) ...
    if (e.target.files && e.target.files[0]) {
        if (currentStream) stopCamera(); // Dừng camera nếu đang chạy
        
        img.onload = () => {
            img.style.display = 'block';
            videoStream.style.display = 'none';
            predict(img);
        };
        img.src = URL.createObjectURL(e.target.files[0]);
    }
});


captureButton.addEventListener('click', () => {
    // ... (Code xử lý chụp ảnh, sau đó gọi predict) ...
    // Vô hiệu hóa nút chụp trong khi xử lý
    captureButton.disabled = true;
    cameraStatus.textContent = 'Đang phân tích...';

    context.drawImage(videoStream, 0, 0, 224, 224);
    
    // Hiển thị ảnh đã chụp và ẩn video stream
    img.src = canvas.toDataURL('image/jpeg');
    img.style.display = 'block';
    videoStream.style.display = 'none';
    
    predict(img);
});

// Hàm mở camera (giữ nguyên)
cameraToggle.addEventListener('click', openCamera);

// Hàm dừng camera (giữ nguyên)
stopButton.addEventListener('click', stopCamera);


// Tải model khi trang được load
initialize();


// ----------------------------------------------------------------
// HÀM XỬ LÝ CAMERA (Giữ nguyên)
// ----------------------------------------------------------------
async function openCamera() {
    if (currentStream) {
        stopCamera();
        cameraToggle.innerHTML = '<span class="camera-btn-text"><i class="material-icons d-block font-size-30">camera_alt</i> Mở Camera</span>';
        cameraContainer.style.display = 'none';
        boxResult.style.display = 'none';
        resultContainer.style.display = 'none';
    } else {
        cameraToggle.innerHTML = '<span class="camera-btn-text"><i class="material-icons d-block font-size-30">videocam_off</i> Đóng Camera</span>';
        cameraContainer.style.display = 'block';
        boxResult.style.display = 'none';
        resultContainer.style.display = 'none';
        img.style.display = 'none'; // Ẩn ảnh đã tải lên
        
        try {
            const constraints = { video: { width: { ideal: 320 }, height: { ideal: 240 }, facingMode: 'environment' } }; 
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
            resultContainer.style.display = 'none';
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
    cameraToggle.innerHTML = '<span class="camera-btn-text"><i class="material-icons d-block font-size-30">camera_alt</i> Mở Camera</span>';
    cameraContainer.style.display = 'none';
}


// ----------------------------------------------------------------
// HÀM XỬ LÝ DARK MODE (Giữ nguyên)
// ----------------------------------------------------------------
modeToggle.addEventListener('click', () => {
    if (body.classList.contains('light-mode')) {
        body.classList.replace('light-mode', 'dark-mode');
        modeToggle.innerHTML = '<i class="material-icons">wb_sunny</i> Chế độ Sáng';
    } else {
        body.classList.replace('dark-mode', 'light-mode');
        modeToggle.innerHTML = '<i class="material-icons">brightness_4</i> Chế độ Tối';
    }
});
