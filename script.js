let model;
let class_indices;
// Biến mới để lưu trữ bản đồ tra cứu chi tiết bệnh (ID -> Object chi tiết)
let disease_lookup_map = {}; 

let fileUpload = document.getElementById('uploadImage');
let img = document.getElementById('image');
let boxResult = document.querySelector('.box-result');
let confidence = document.querySelector('.confidence');
let pconf = document.querySelector('.box-result p');
let modeToggle = document.getElementById('modeToggle');
let body = document.body;

// Khai báo biến DOM mới cho chi tiết bệnh
let diseaseDetailsContainer = document.getElementById('diseaseDetails');


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
   
    let indices = {};
    // Đảm bảo disease_lookup_map được reset
    disease_lookup_map = {};

    // Giả sử data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet là mảng chứa các đối tượng bệnh
    if (data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet && Array.isArray(data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet)) {
        data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet.forEach(item => {
            // Sử dụng Mã_ID làm khóa tra cứu (giả sử trùng với index của model)
            const id = item.Mã_ID; 
            if (id !== undefined) {
                // Bản đồ tra cứu Tên_Bệnh (dùng cho pred_class)
                indices[id] = item.Tên_Bệnh; 
                // Bản đồ tra cứu chi tiết bệnh (dùng cho diseaseDetailsContainer)
                disease_lookup_map[id] = item; 
            }
        });
    } else {
        // Xử lý trường hợp JSON có cấu trúc cũ (chỉ chứa Tên Bệnh) hoặc cấu trúc Object tối ưu
        for (const key in data) {
            // Trường hợp JSON Object tối ưu: data[key] là Object chi tiết bệnh, key là ID
            if (typeof data[key] === 'object' && data[key] !== null && data[key].Tên_Bệnh) {
                 indices[key] = data[key].Tên_Bệnh;
                 disease_lookup_map[key] = data[key];
            } 
            // Trường hợp JSON cũ (chỉ có index -> name): data[key] là index, key là Tên Bệnh
            else if (typeof data[key] === 'string' && /^\d+$/.test(data[key])) { 
                 indices[data[key]] = key;
                 // Không có chi tiết bệnh để tra cứu trong trường hợp này
            }
        }
    }
    
    // Lưu bản đồ tra cứu Tên_Bệnh vào biến global
    class_indices = indices;
    return indices; 
}


async function initialize() {
    let status = document.querySelector('.init_status')
    status.innerHTML = 'Đang tải mô hình .... <span class="fa fa-spinner fa-spin"></span>'
    model = await tf.loadLayersModel('./tensorflowjs-model/model.json');
    status.innerHTML = 'Tải mô hình thành công <span class="fa fa-check"></span>'
    boxResult.style.display = 'block'
    // Tải dữ liệu JSON ngay sau khi model được tải
    await fetchData();
}

async function predict() {

    // Khởi tạo model và tải dữ liệu JSON nếu chưa có
    if (!model) {
        await initialize();
    }
    
    // Ẩn chi tiết bệnh cũ
    diseaseDetailsContainer.innerHTML = '';

    let offset = tf.scalar(255)
    

    let tensorImg =   tf.browser.fromPixels(img).resizeNearestNeighbor([224,224]).toFloat().expandDims();
    let tensorImg_scaled = tensorImg.div(offset)
    
    let prediction = await model.predict(tensorImg_scaled).data();

    onPredict(tf.tensor(prediction));
    
}

function onPredict(pred) {
    let index = pred.argMax().dataSync()[0];
    let confidenceValue = pred.dataSync()[index];
    
    // 1. Hiển thị Tên Bệnh và Độ chính xác
    let pred_class_name = class_indices[index] || 'Không xác định'; 
    document.querySelector('.pred_class').innerHTML = pred_class_name;
    document.querySelector('.inner').innerHTML = `${parseFloat(confidenceValue * 100).toFixed(2)}%`;

    progressBar.set(0);
    progressBar.animate(confidenceValue - 0.005); 

    pconf.style.display = 'block';

    confidence.innerHTML = Math.round(confidenceValue * 100);

    document.querySelector('.init_status').innerHTML = '';

    // 2. LOGIC CẢI TIẾN: Lấy và hiển thị thông tin chi tiết về bệnh
    const diseaseDetails = disease_lookup_map[index];
    if (diseaseDetails) {
        displayDiseaseDetails(diseaseDetails);
    } else {
        diseaseDetailsContainer.innerHTML = '<p class="error-msg">⚠️ Không tìm thấy thông tin chi tiết về phác đồ quản lý bệnh này.</p>';
        console.error(`Không tìm thấy chi tiết bệnh cho Mã ID: ${index}`);
    }
}

// Hàm mới để định dạng và hiển thị thông tin chi tiết từ JSON
function displayDiseaseDetails(details) {
    // Hàm hỗ trợ render danh sách phác đồ con (ví dụ: III_Chiến_lược_Kiểm_soát_Hóa_học.Phác_đồ_Giai_đoạn_Cây)
    const renderPhacDoGiaiDoan = (steps) => {
        if (!Array.isArray(steps) || steps.length === 0) {
            return '<li>Thông tin phác đồ cụ thể đang được cập nhật.</li>';
        }
        return steps.map(step => 
            `<li><strong>${step.Giai_đoạn || 'Giai đoạn N/A'}:</strong> ${step.Hoạt_chất_Đề_xuất || 'N/A'} (Nhóm ${step.Nhóm_FRAC_IRAC || 'N/A'}). 
            <span class="note">Lưu ý: ${step.Lưu_ý_Ứng_dụng || 'Không có lưu ý đặc biệt.'}</span></li>`
        ).join('');
    };

    let html = `
        <div class="details-section">
            <h3 class="details-title">🌿 PHÁC ĐỒ QUẢN LÝ TỔNG HỢP (IPM)</h3>
            <p><strong>Bệnh:</strong> ${details.Tên_Bệnh || 'N/A'}</p>
            <p><strong>Mã ID:</strong> ${details.Mã_ID || 'N/A'}</p>
            <p><strong>Phân loại:</strong> ${details.Phân_loại || 'Đang cập nhật'}</p>
            <hr>
            
            <h4>1. Tác nhân, Chu kỳ và Điều kiện</h4>
            <div class="content-box">
                <p><strong>Tác nhân Sinh học:</strong> ${details.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Tác_nhân_Sinh_học || 'N/A'}</p>
                <p><strong>Cơ chế Lây lan:</strong> ${details.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Cơ_chế_Lây_lan || 'N/A'}</p>
                <p><strong>Nhiệt độ/Thời điểm tối ưu:</strong> ${details.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Nhiệt_độ_Thời_điểm_tối_ưu || 'N/A'}</p>
            </div>
            
            <h4>2. Chiến lược Kiểm soát Văn hóa và Vật lý</h4>
            <div class="content-box">
                <p>${details.II_Chiến_lược_Kiểm_soát_Văn_hóa_và_Vật_lý.Nguyên_tắc_Cơ_bản || 'Chưa có dữ liệu chi tiết.'}</p>
            </div>
            
            <h4>3. Chiến lược Kiểm soát Hóa học (Ưu tiên luân phiên nhóm thuốc)</h4>
            <div class="content-box">
                <p><strong>Nguyên tắc FRAC/IRAC:</strong> ${details.III_Chiến_lược_Kiểm_soát_Hóa_học.Nguyên_tắc_FRAC_IRAC || 'N/A'}</p>
                <ul>
                    ${renderPhacDoGiaiDoan(details.III_Chiến_lược_Kiểm_soát_Hóa_học.Phác_đồ_Giai_đoạn_Cây)}
                </ul>
            </div>

            <h4>4. Giải pháp Sinh học và Thân thiện Môi trường</h4>
            <div class="content-box">
                <p>${details.IV_Giải_pháp_Sinh_học_và_Thân_thiện_Môi_trường.Các_Hoạt_chất_Sinh_học_Đề_xuất || 'Chưa có dữ liệu chi tiết.'}</p>
            </div>

        </div>
    `;
    diseaseDetailsContainer.innerHTML = html;
}


fileUpload.addEventListener('change', function(e){

    stopCamera();
    cameraContainer.style.display = 'none';

    let uploadedImage = e.target.value
    if (uploadedImage){
        document.getElementById("blankFile-1").innerHTML = uploadedImage.replace("C:\\fakepath\\","")
        document.getElementById("choose-text-1").innerText = "Đổi Ảnh Đã Chọn"
        document.querySelector(".success-1").style.display = "inline-block"

    
        document.querySelector(".success-1 i").style.border = "1px solid limegreen"
        document.querySelector(".success-1 i").style.color = "limegreen"
        
    }
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
    if (cameraContainer.style.display === 'flex') {
        stopCamera();
        cameraContainer.style.display = 'none';
    } else {
        cameraContainer.style.display = 'flex';
        startCamera();
    }
});

stopButton.addEventListener('click', function() {
    stopCamera();
    cameraContainer.style.display = 'none';
});

captureButton.addEventListener('click', function() {
    if (currentStream) {

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
        diseaseDetailsContainer.innerHTML = ''; // Ẩn kết quả khi mở camera
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
            diseaseDetailsContainer.innerHTML = ''; // Ẩn kết quả khi mở camera
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

// Gọi initialize() để tải model và JSON khi trang được tải lần đầu
// initialize(); 
// Đặt initialize trong predict để đảm bảo model được tải khi cần thiết lần đầu
