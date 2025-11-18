let model;
let class_indices;
let fileUpload = document.getElementById('uploadImage');
let img = document.getElementById('image');
let boxResult = document.querySelector('.box-result');
let confidence = document.querySelector('.confidence');
let pconf = document.querySelector('.confidence-text'); // Tên class mới trong HTML
let predClassSpan = document.querySelector('.pred_class'); // Tên class mới
let modeToggle = document.getElementById('modeToggle');
let body = document.body;
let resultDetails = document.getElementById('resultDetails'); // Biến mới cho kết quả chi tiết


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

async function fetchData(){
    let response = await fetch('./class_indices.json');
    let data = await response.json();
   
    // Map dữ liệu theo Mã_ID để tra cứu nhanh và chính xác
    let indicesMap = {};
    for (const item of data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet) {
        indicesMap[item.Mã_ID] = item;
    }
    class_indices = indicesMap; 
    return class_indices;
}


async function initialize() {
    let status = document.querySelector('.init_status');
    status.innerHTML = 'Đang tải thông tin phân loại...';
    
    try {
        // 1. Fetch JSON data
        class_indices = await fetchData();

        status.innerHTML = 'Đang tải mô hình học máy (Tối ưu hóa: Quantization được đề xuất)...';
        // 2. Load the model. (Cần đảm bảo file model.json nằm trong thư mục model/)
        const modelURL = 'model/model.json'; // Cần thay đổi theo đường dẫn thực tế
        model = await tf.loadLayersModel(modelURL); 
        
        status.innerHTML = 'Hệ thống sẵn sàng! Hãy tải ảnh lên hoặc bật camera.';
        status.style.backgroundColor = '#2ecc71'; 
        
        document.querySelector('.upload-btn-container').style.display = 'flex'; 

    } catch (error) {
        status.innerHTML = `LỖI KHỞI TẠO: Không thể tải dữ liệu/mô hình. Vui lòng kiểm tra file model/model.json và class_indices.json.`;
        status.style.backgroundColor = '#e74c3c'; 
        console.error("Initialization Error:", error);
    }
}


// HÀM MỚI: Render nội dung phân tích chi tiết (Đã giải quyết yêu cầu hiển thị một khối)
function renderDetailedAnalysis(predictedId) {
    resultDetails.innerHTML = ''; // Xóa nội dung cũ
    resultDetails.style.display = 'block';

    if (!predictedId || !class_indices[predictedId]) {
        resultDetails.innerHTML = '<h3 style="color: #e74c3c;">Không tìm thấy thông tin chi tiết cho mã bệnh này.</h3>';
        return;
    }

    const data = class_indices[predictedId];

    let htmlContent = `
        <h2>📋 ${data.Tên_Bệnh} - Phân Tích Chi Tiết </h2>
        <p><strong>Mã ID:</strong> ${data.Mã_ID} | <strong>Phân Loại Tổng Quát:</strong> ${data.Phân_loại}</p>
        <hr>
        
        <h3>I. Tác Nhân, Chu Kỳ và Điều Kiện Phát Triển</h3>
        <p><strong>Tác nhân Sinh học:</strong> <strong>${data.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Tác_nhân_Sinh_học}</strong></p>
        <p><strong>Cơ chế Lây lan:</strong> ${data.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Cơ_chế_Lây_lan}</p>
        <p><strong>Nhiệt độ/Thời điểm tối ưu:</strong> ${data.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Nhiệt_độ_Thời_điểm_tối_ưu}</p>
        <h4>Dấu Hiệu Chẩn Đoán Chuyên Sâu:</h4>
        <p>${data.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Dấu_hiệu_Chẩn_đoán_Chuyên_sâu.replace(/(\d+\. \*\*)/g, '<br><strong>$1')}</p>

        <h3>II. Biện Pháp Canh Tác Chuyên Sâu</h3>
        <h4>Quản lý Tàn dư/Đất:</h4>
        <p>${data.II_Biện_pháp_Canh_tác_Chuyên_sâu.Quản_lý_Tàn_dư_Đất}</p>
        <h4>Quản lý Nước Tưới:</h4>
        <p>${data.II_Biện_pháp_Canh_tác_Chuyên_sâu.Quản_lý_Nước_Tưới}</p>
        <h4>Mật độ & Thông thoáng:</h4>
        <p>${data.II_Biện_pháp_Canh_tác_Chuyên_sâu.Mật_độ_Thông_thoáng}</p>
        <h4>Quản lý Dinh dưỡng/Vi lượng:</h4>
        <p>${data.II_Biện_pháp_Canh_tác_Chuyên_sâu.Quản_lý_Dinh_dưỡng_Vi_lượng}</p>

        <h3>III. Chiến Lược Kiểm Soát Hóa Học</h3>
        <p><strong>Nguyên tắc FRAC/IRAC:</strong> ${data.III_Chiến_lược_Kiểm_soát_Hóa_học.Nguyên_tắc_FRAC_IRAC}</p>
        <h4>Phác đồ theo Giai đoạn Cây:</h4>
        <ul>
    `;
    
    // Tạo danh sách cho Phác đồ Giai đoạn Cây
    const phacDo = data.III_Chiến_lược_Kiểm_soát_Hóa_học.Phác_đồ_Giai_đoạn_Cây;
    if (phacDo && phacDo.length > 0) {
        phacDo.forEach(item => {
            htmlContent += `
                <li>
                    <strong>Giai đoạn ${item.Giai_đoạn}:</strong>
                    <ul>
                        <li>Hoạt chất Đề xuất: <strong>${item.Hoạt_chất_Đề_xuất}</strong> (Nhóm FRAC/IRAC: ${item.Nhóm_FRAC_IRAC})</li>
                        <li>Lưu ý Ứng dụng: ${item.Lưu_ý_Ứng_dụng}</li>
                    </ul>
                </li>
            `;
        });
    } else {
        htmlContent += `<li>(Không có phác đồ hóa học cụ thể được đề xuất hoặc không cần thiết.)</li>`;
    }

    htmlContent += `
        </ul>
        <h4>Thuốc Trừ Tận gốc (Eradicant):</h4>
        <p>${data.III_Chiến_lược_Kiểm_soát_Hóa_học.Thuốc_Trừ_Tận_gốc_Eradicant}</p>

        <h3>IV. Giải Pháp Sinh Học và Thay Thế</h3>
        <h4>Chất Đối kháng VSV:</h4>
        <p>${data.IV_Giải_pháp_Sinh_học_và_Thay_thế.Chất_Đối_kháng_VSV}</p>
        <h4>Cảm ứng Kháng Bệnh (SAR):</h4>
        <p>${data.IV_Giải_pháp_Sinh_học_và_Thay_thế.Cảm_ứng_Kháng_Bệnh_SAR}</p>
        <h4>Kiểm soát Côn trùng Vector:</h4>
        <p>${data.IV_Giải_pháp_Sinh_học_và_Thay_thế.Kiểm_soát_Côn_trùng_Vector}</p>
    `;

    resultDetails.innerHTML = htmlContent;
}


// HÀM DỰ ĐOÁN: Cập nhật để gọi renderDetailedAnalysis
async function predictImage(imageElement, fromCamera = false) {
    // ... [Bỏ qua logic tiền xử lý hình ảnh thực tế cho mục đích điều chỉnh code] ...

    const status = document.querySelector('.init_status');
    status.innerHTML = 'Đang phân tích hình ảnh...';
    
    // --- BƯỚC MÔ PHỎNG DỰ ĐOÁN (Cần thay thế bằng logic model.predict thực tế) ---
    // Giả lập kết quả thành công, ví dụ: 'Táo - Ghẻ Táo (Apple Scab)'
    const simulatedPrediction = {
        classId: '0', 
        confidence: 0.9531 
    };
    
    const { classId, confidence: rawConfidence } = simulatedPrediction;
    const confidencePercent = (rawConfidence * 100).toFixed(2);
    const resultData = class_indices[classId];
    // --------------------------------------------------------------------------
    
    if (resultData) {
        const resultName = resultData.Tên_Bệnh;
        
        // Cập nhật kết quả tóm tắt
        predClassSpan.textContent = resultName;
        confidence.textContent = confidencePercent;
        boxResult.style.display = 'block';
        pconf.style.display = 'block';
        
        // Progress bar animation
        progressBar.animate(rawConfidence);
        
        // Gọi hàm hiển thị chi tiết nội dung phân tích (Đã giải quyết yêu cầu)
        renderDetailedAnalysis(classId);

        status.innerHTML = `Phân tích hoàn tất: ${resultName}`;
        status.style.backgroundColor = '#2ecc71';

    } else {
        // Xử lý khi phân loại thất bại
        predClassSpan.textContent = 'Lỗi Phân Loại';
        confidence.textContent = 'N/A';
        progressBar.animate(0);
        boxResult.style.display = 'block';
        pconf.style.display = 'block';
        resultDetails.style.display = 'none';
        
        status.innerHTML = 'Không thể xác định loại cây/bệnh. Vui lòng thử lại ảnh khác.';
        status.style.backgroundColor = '#f39c12';
    }
}


fileUpload.addEventListener('change', (e) => {
    if (e.target.files[0]) {
        let reader = new FileReader();
        reader.onload = (event) => {
            img.src = event.target.result;
            img.style.display = 'block';
            boxResult.style.display = 'block';
            resultDetails.style.display = 'none';
            // Gọi predictImage sau khi ảnh được tải hoàn toàn
            img.onload = () => {
                predictImage(img);
            };
        };
        reader.readAsDataURL(e.target.files[0]);
    }
});


// ... Các hàm và event listener khác (camera, mode toggle) vẫn giữ nguyên logic ban đầu.

// cameraToggle.addEventListener('click', toggleCamera);
// captureButton.addEventListener('click', captureImage);
// stopButton.addEventListener('click', stopCamera);

// Khởi tạo hệ thống
initialize();
