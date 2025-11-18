let model;
let class_indices;
let fileUpload = document.getElementById('uploadImage');
let img = document.getElementById('image');
let boxResult = document.querySelector('.box-result');
let confidenceSpan = document.querySelector('.confidence'); // Đổi tên biến để tránh nhầm lẫn
let predClassSpan = document.querySelector('.pred_class'); // Thêm biến cho tên bệnh
let modeToggle = document.getElementById('modeToggle');
let body = document.body;

let cameraToggle = document.getElementById('cameraToggle');
let videoStream = document.getElementById('videoStream');
let captureButton = document.getElementById('captureButton');
let stopButton = document.getElementById('stopButton');
let cameraStatus = document.getElementById('cameraStatus');
let canvas = document.getElementById('canvas');
let context = canvas.getContext('2d');
let currentStream;

// KHU VỰC MỚI CHO PHÂN TÍCH CHI TIẾT
let analysisContainer = document.getElementById('analysisContainer');
let analysisContent = document.getElementById('analysisContent');

let progressBar =
    new ProgressBar.Circle('#progress', {
    color: 'limegreen',
    strokeWidth: 10,
    duration: 2000,
    easing: 'easeInOut'
});

async function fetchData(){
    let status = document.querySelector('.init_status');
    status.innerHTML = 'Đang tải thông tin bệnh cây (JSON)...';

    try {
        let response = await fetch('./class_indices.json');
        if (!response.ok) {
            throw new Error(`Lỗi tải file JSON: ${response.statusText}`);
        }
        let data = await response.json();
        
        // Chuyển đổi dữ liệu JSON thành Map tiện lợi hơn cho việc tìm kiếm
        class_indices = new Map();
        for (const item of data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet) {
            class_indices.set(item.Mã_ID, item);
        }
        return class_indices;
    } catch (error) {
        status.innerHTML = `Lỗi: ${error.message}. Không thể tải phác đồ quản lý.`;
        console.error("Lỗi tải JSON:", error);
        return null;
    }
}


async function initialize() {
    let status = document.querySelector('.init_status');
    status.innerHTML = 'Đang khởi tạo hệ thống...';
    try {
        // Tải JSON trước (Đã tối ưu bằng await trong fetchData)
        class_indices = await fetchData();
        if (!class_indices) return;

        // Tải Model (Tối ưu bằng await)
        status.innerHTML = 'Đang tải model AI...';
        // Giả định model nằm ở './model/model.json'. Thay đổi nếu đường dẫn khác.
        model = await tf.loadLayersModel('./model/model.json'); 
        status.innerHTML = 'Hệ thống sẵn sàng! Hãy tải ảnh hoặc mở Camera.';
        status.style.backgroundColor = '#4caf50';
        boxResult.style.display = 'block';

    } catch (error) {
        status.innerHTML = `Lỗi: Không thể tải model. Vui lòng kiểm tra lại file model/model.json. Chi tiết: ${error.message}`;
        status.style.backgroundColor = 'red';
        console.error("Lỗi khởi tạo:", error);
    }
}

// Hàm chuyển đổi ảnh thành tensor (đảm bảo chính xác)
function preprocessImage(imageElement) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([224, 224]) // Đảm bảo kích thước chuẩn
            .toFloat()
            .div(tf.scalar(255.0)) // Chuẩn hóa
            .expandDims();
        return tensor;
    });
}

// Hàm dự đoán
async function predict(imageElement) {
    if (!model || !class_indices) {
        alert('Hệ thống chưa được khởi tạo hoàn chỉnh. Vui lòng đợi.');
        return;
    }

    progressBar.animate(0); // Reset progress bar
    analysisContainer.style.display = 'none'; // Ẩn kết quả cũ

    let status = document.querySelector('.init_status');
    status.innerHTML = 'Đang phân tích hình ảnh...';
    status.style.backgroundColor = '#2196F3'; // Màu xanh dương cho trạng thái phân tích

    try {
        const tensor = preprocessImage(imageElement);
        const predictions = await model.predict(tensor).data();
        const output = Array.from(predictions);

        // Tìm lớp có xác suất cao nhất
        const maxConfidence = Math.max(...output);
        const classIndex = output.indexOf(maxConfidence).toString(); // Chuyển sang chuỗi để so sánh với Mã_ID

        const confidencePercentage = (maxConfidence * 100).toFixed(2);

        // Cập nhật giao diện kết quả cơ bản
        confidenceSpan.textContent = confidencePercentage;
        progressBar.animate(maxConfidence);

        // Lấy thông tin chi tiết từ class_indices Map
        const diseaseInfo = class_indices.get(classIndex);
        
        if (diseaseInfo) {
            predClassSpan.textContent = diseaseInfo.Tên_Bệnh;
            status.innerHTML = 'Phân tích hoàn tất!';
            status.style.backgroundColor = '#4caf50';
            
            // HIỂN THỊ NỘI DUNG PHÂN TÍCH CHI TIẾT
            renderAnalysisContent(diseaseInfo);

        } else {
            predClassSpan.textContent = 'Lỗi: Không tìm thấy thông tin chi tiết cho Mã ID ' + classIndex;
            status.innerHTML = 'Phân tích hoàn tất, nhưng thiếu dữ liệu chi tiết!';
            status.style.backgroundColor = 'orange';
            analysisContainer.style.display = 'none';
        }
        
        if (maxConfidence < 0.6) { // Thêm cảnh báo nếu độ chính xác thấp
             status.innerHTML += ' (Độ chính xác thấp, kết quả có thể không đáng tin cậy!)';
             status.style.backgroundColor = 'red';
        }


    } catch (error) {
        status.innerHTML = `Lỗi trong quá trình dự đoán: ${error.message}`;
        status.style.backgroundColor = 'red';
        console.error("Lỗi dự đoán:", error);
    }
}

// HÀM MỚI: RENDER NỘI DUNG PHÂN TÍCH CHI TIẾT (MỘT KHỐI)
function renderAnalysisContent(diseaseInfo) {
    analysisContainer.style.display = 'block';
    let htmlContent = `
        <h3>${diseaseInfo.Tên_Bệnh}</h3>
        <p><strong>Phân loại:</strong> ${diseaseInfo.Phân_loại}</p>
        <hr>
    `;

    // Phần I: Tác nhân, Chu kỳ và Điều kiện
    if (diseaseInfo.I_Tác_nhân_Chu_kỳ_và_Điều_kiện) {
        const part = diseaseInfo.I_Tác_nhân_Chu_kỳ_và_Điều_kiện;
        htmlContent += `
            <h4>I. Tác nhân, Chu kỳ và Điều kiện</h4>
            <p><strong>Tác nhân Sinh học:</strong> ${part.Tác_nhân_Sinh_học}</p>
            <p><strong>Cơ chế Lây lan:</strong> ${part.Cơ_chế_Lây_lan}</p>
            <p><strong>Nhiệt độ/Thời điểm tối ưu:</strong> ${part.Nhiệt_độ_Thời_điểm_tối_ưu}</p>
            <p><strong>Dấu hiệu Chẩn đoán Chuyên sâu:</strong> ${part.Dấu_hiệu_Chẩn_đoán_Chuyên_sâu}</p>
            <hr>
        `;
    }
    
    // Phần II: Chiến lược Kiểm soát Văn hóa & Sinh học
    if (diseaseInfo.II_Chiến_lược_Kiểm_soát_Văn_hóa_và_Dinh_dưỡng) {
        const part = diseaseInfo.II_Chiến_lược_Kiểm_soát_Văn_hóa_và_Dinh_dưỡng;
        htmlContent += `
            <h4>II. Chiến lược Kiểm soát Văn hóa & Dinh dưỡng</h4>
            <p><strong>Nguyên tắc Quản lý Văn hóa:</strong> ${part.Nguyên_tắc_Quản_lý_Văn_hóa}</p>
            <p><strong>Thực hành Cụ thể:</strong> ${part.Thực_hành_Cụ_thể}</p>
            <p><strong>Quản lý Dinh dưỡng:</strong> ${part.Quản_lý_Dinh_dưỡng}</p>
            <hr>
        `;
    }

    // Phần III: Chiến lược Kiểm soát Hóa học
    if (diseaseInfo.III_Chiến_lược_Kiểm_soát_Hóa_học) {
        const part = diseaseInfo.III_Chiến_lược_Kiểm_soát_Hóa_học;
        htmlContent += `
            <h4>III. Chiến lược Kiểm soát Hóa học</h4>
            <p><strong>Nguyên tắc FRAC/IRAC:</strong> ${part.Nguyên_tắc_FRAC_IRAC}</p>
            <p><strong>Thuốc Trừ Tận gốc (Eradicant):</strong> ${part.Thuốc_Trừ_Tận_gốc_Eradicant}</p>
            
            <h5>Phác đồ Giai đoạn Cây:</h5>
            <ul>
        `;
        // Hiển thị danh sách phác đồ theo giai đoạn
        if (part.Phác_đồ_Giai_đoạn_Cây && part.Phác_đồ_Giai_đoạn_Cây.length > 0) {
            part.Phác_đồ_Giai_đoạn_Cây.forEach(stage => {
                htmlContent += `
                    <li>
                        <strong>Giai đoạn ${stage.Giai_đoạn}:</strong> ${stage.Hoạt_chất_Đề_xuất} (Nhóm: ${stage.Nhóm_FRAC_IRAC}) - <em>Lưu ý: ${stage.Lưu_ý_Ứng_dụng}</em>
                    </li>
                `;
            });
        } else {
             htmlContent += `<li>Không có phác đồ hóa học cụ thể được đề xuất.</li>`;
        }
        htmlContent += `</ul><hr>`;
    }
    
    // Phần IV: Giải pháp Sinh học và Kháng sinh
    if (diseaseInfo.IV_Giải_pháp_Sinh_học_và_Kháng_sinh) {
        const part = diseaseInfo.IV_Giải_pháp_Sinh_học_và_Kháng_sinh;
        htmlContent += `
            <h4>IV. Giải pháp Sinh học và Kháng sinh</h4>
            <p><strong>Sản phẩm Sinh học Đề xuất:</strong> ${part.Sản_phẩm_Sinh_học_Đề_xuất}</p>
            <p><strong>Nguyên tắc Ứng dụng:</strong> ${part.Nguyên_tắc_Ứng_dụng}</p>
            <hr>
        `;
    }

    // Gán nội dung đã tạo vào container
    analysisContent.innerHTML = htmlContent;
}


// Xử lý sự kiện tải ảnh
fileUpload.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        const reader = new FileReader();
        
        reader.onload = function(event) {
            img.onload = function() {
                // Hiển thị ảnh
                img.style.display = 'block';
                videoStream.style.display = 'none'; // Ẩn camera nếu đang mở
                
                // Bắt đầu dự đoán
                predict(img);
            }
            img.src = event.target.result;
        }
        reader.readAsDataURL(file);
    }
});

// Xử lý sự kiện chụp ảnh từ camera
captureButton.addEventListener('click', () => {
    // Vẽ khung hình hiện tại của video lên canvas
    context.drawImage(videoStream, 0, 0, canvas.width, canvas.height);
    
    // Ẩn video, hiện ảnh đã chụp (từ canvas)
    videoStream.style.display = 'none';
    img.style.display = 'block';
    img.src = canvas.toDataURL('image/png'); // Dùng data URL từ canvas làm nguồn ảnh
    
    // Dừng camera sau khi chụp
    stopCamera(); 
    
    // Bắt đầu dự đoán từ canvas (hoặc ảnh đã được gán src)
    // Dùng img đã gán src từ canvas
    predict(img); 
});


// Logic Camera (giữ nguyên, đảm bảo chức năng)
// ... (Phần logic camera của bạn) ...
let cameraContainer = document.getElementById('cameraContainer');
let videoElement = document.getElementById('videoStream');
let stopButton = document.getElementById('stopButton');


cameraToggle.addEventListener('click', () => {
    if (cameraContainer.style.display === 'block') {
        stopCamera();
        cameraContainer.style.display = 'none';
        cameraToggle.innerHTML = '<span class="upload-btn"><i class="material-icons d-block font-size-30">photo_camera</i> Mở Camera</span>';
    } else {
        startCamera();
        cameraContainer.style.display = 'block';
        cameraToggle.innerHTML = '<span class="upload-btn"><i class="material-icons d-block font-size-30">photo_camera</i> Đóng Camera</span>';
    }
});

stopButton.addEventListener('click', stopCamera);

async function startCamera() {
    if (currentStream) {
        stopCamera();
    }
    try {
        currentStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: 320, height: 240 } });
        videoStream.srcObject = currentStream;
        videoStream.play();
        cameraStatus.textContent = 'Camera đã sẵn sàng. Hãy chụp ảnh.';
        captureButton.disabled = false;
        videoStream.style.display = 'block';
        captureButton.style.display = 'block';
        stopButton.style.display = 'block';
        img.style.display = 'none';
        boxResult.style.display = 'block';
        analysisContainer.style.display = 'none'; // Ẩn phân tích cũ
    } catch (error) {
        cameraStatus.textContent = `Lỗi truy cập camera: ${error.name}. Vui lòng đảm bảo camera được phép sử dụng.`;
        captureButton.disabled = true;
        videoStream.style.display = 'none';
        captureButton.style.display = 'none';
        stopButton.style.display = 'none';
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
    videoStream.style.display = 'none';
    captureButton.style.display = 'none';
    stopButton.style.display = 'none';
}

// Logic chuyển đổi chế độ Sáng/Tối
modeToggle.addEventListener('click', () => {
    if (body.classList.contains('light-mode')) {
        body.classList.replace('light-mode', 'dark-mode');
        modeToggle.innerHTML = '<i class="material-icons">wb_sunny</i> Chế độ Sáng';
    } else {
        body.classList.replace('dark-mode', 'light-mode');
        modeToggle.innerHTML = '<i class="material-icons">brightness_4</i> Chế độ Tối';
    }
});

// Khởi tạo hệ thống khi tải trang
window.onload = initialize;
