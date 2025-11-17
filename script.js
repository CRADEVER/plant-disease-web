let model;
let class_indices;
let fileUpload = document.getElementById('uploadImage');
let img = document.getElementById('image');
let boxResult = document.querySelector('.box-result');
let confidence = document.querySelector('.confidence');
let pconf = document.querySelector('.box-result > p'); // Selector chính xác hơn
let modeToggle = document.getElementById('modeToggle');
let body = document.body;

// Camera elements
let cameraToggle = document.getElementById('cameraToggle');
let cameraContainer = document.getElementById('cameraContainer');
let videoStream = document.getElementById('videoStream');
let captureButton = document.getElementById('captureButton');
let stopButton = document.getElementById('stopButton');
let cameraStatus = document.getElementById('cameraStatus');
let canvas = document.getElementById('canvas');
let context = canvas.getContext('2d');
let currentStream;

// Các element hiển thị kết quả chi tiết (MỚI)
let predClassification = document.getElementById('pred_classification');
let predSigns = document.getElementById('pred_signs');
let predFarming = document.getElementById('pred_farming');
let predChemical = document.getElementById('pred_chemical');
let predBiological = document.getElementById('pred_biological');
let infoContainer = document.getElementById('infoContainer');


let progressBar =
    new ProgressBar.Circle('#progress', {
    color: 'limegreen',
    strokeWidth: 10,
    duration: 2000,
    easing: 'easeInOut'
});

/**
 * CẬP NHẬT fetchData
 * Tải file JSON mới và trả về mảng "Phac_do_Quan_Ly_Tong_Hop_Chi_tiet"
 */
async function fetchData(){
    // Thêm cache-buster để đảm bảo tải file JSON mới nhất
    let response = await fetch('./class_indices.json?v=' + new Date().getTime());
    let data = await response.json();
    // Trả về mảng dữ liệu chi tiết
    return data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet;
}


async function initialize() {
    let status = document.querySelector('.init_status')
    status.innerHTML = 'Đang tải mô hình .... <span class="fa fa-spinner fa-spin"></span>'
    model = await tf.loadLayersModel('./tensorflowjs-model/model.json');
    status.innerHTML = 'Tải mô hình thành công <span class="fa fa-check"></span>'
    boxResult.style.display = 'block'
}

/**
 * CẬP NHẬT predict
 * Tìm kiếm trong mảng JSON mới và hiển thị dữ liệu chi tiết
 */
async function predict() {

    await initialize(); 
    let offset = tf.scalar(255)
    
    let tensorImg =   tf.browser.fromPixels(img).resizeNearestNeighbor([224,224]).toFloat().expandDims();
    let tensorImg_scaled = tensorImg.div(offset)
    
    let prediction = await model.predict(tensorImg_scaled).data();

    // allData bây giờ là MẢNG các phác đồ
    fetchData().then((allData)=>
        {
            predicted_class = tf.argMax(prediction)
            
            // class_idx là chỉ số do mô hình trả về (ví dụ: 0, 1, 2...)
            class_idx = Array.from(predicted_class.dataSync())[0]

            // Tìm đối tượng bệnh trong mảng dựa trên Mã_ID (so sánh chuỗi)
            const result = allData.find(item => item.Mã_ID == class_idx.toString());

            if (result) {
                // Hiển thị container thông tin
                infoContainer.style.display = "block";

                // 1. Tên Bệnh & Phân loại
                document.querySelector('.pred_class').innerHTML = result.Tên_Bệnh || "Không rõ";
                predClassification.innerHTML = `<strong>Phân loại:</strong> ${result.Phân_loại || "N/A"}`;

                // 2. Dấu hiệu & Tác nhân
                let signsText = `
                    <strong>Tác nhân:</strong> ${result.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Tác_nhân_Sinh_học || 'N/A'}<br>
                    <strong>Cơ chế lây lan:</strong> ${result.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Cơ_chế_Lây_lan || 'N/A'}<br>
                    <strong>Điều kiện tối ưu:</strong> ${result.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Nhiệt_độ_Thời_điểm_tối_ưu || 'N/A'}<br>
                    <strong>Chẩn đoán:</strong> ${result.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Dấu_hiệu_Chẩn_đoán_Chuyên_sâu || 'N/A'}
                `;
                predSigns.innerHTML = signsText.replace(/  +/g, ' ').trim();

                // 3. Biện pháp Canh tác
                let farmingText = `
                    <strong>Quản lý Tàn dư:</strong> ${result.II_Biện_pháp_Canh_tác_Chuyên_sâu.Quản_lý_Tàn_dư_Đất || 'N/A'}<br>
                    <strong>Quản lý Nước tưới:</strong> ${result.II_Biện_pháp_Canh_tác_Chuyên_sâu.Quản_lý_Nước_Tưới || 'N/A'}<br>
                    <strong>Thông thoáng:</strong> ${result.II_Biện_pháp_Canh_tác_Chuyên_sâu.Mật_độ_Thông_thoáng || 'N/A'}<br>
                    <strong>Dinh dưỡng:</strong> ${result.II_Biện_pháp_Canh_tác_Chuyên_sâu.Quản_lý_Dinh_dưỡng_Vi_lượng || 'N/A'}
                `;
                predFarming.innerHTML = farmingText.replace(/  +/g, ' ').trim();

                // 4. Chiến lược Hóa học
                let chemicalText = `<strong>Nguyên tắc:</strong> ${result.III_Chiến_lược_Kiểm_soát_Hóa_học.Nguyên_tắc_FRAC_IRAC || 'N/A'}<br>`;
                if (result.III_Chiến_lược_Kiểm_soát_Hóa_học.Phác_đồ_Giai_đoạn_Cây) {
                    chemicalText += "<strong>Phác đồ giai đoạn:</strong><br>";
                    result.III_Chiến_lược_Kiểm_soát_Hóa_học.Phác_đồ_Giai_đoạn_Cây.forEach(stage => {
                        chemicalText += `&nbsp;&nbsp;&nbsp;• <em>${stage.Giai_đoạn}:</em> ${stage.Hoạt_chất_Đề_xuất} (Nhóm: ${stage.Nhóm_FRAC_IRAC}). ${stage.Lưu_ý_Ứng_dụng}<br>`;
                    });
                }
                chemicalText += `<strong>Thuốc trị tận gốc:</strong> ${result.III_Chiến_lược_Kiểm_soát_Hóa_học.Thuốc_Trừ_Tận_gốc_Eradicant || 'N/A'}`;
                predChemical.innerHTML = chemicalText.replace(/  +/g, ' ').trim();

                // 5. Giải pháp Sinh học
                let bioText = `
                    <strong>Chất đối kháng:</strong> ${result.IV_Giải_pháp_Sinh_học_và_Thay_thế.Chất_Đối_kháng_VSV || 'N/A'}<br>
                    <strong>Cảm ứng kháng bệnh (SAR):</strong> ${result.IV_Giải_pháp_Sinh_học_và_Thay_thế.Cảm_ứng_Kháng_Bệnh_SAR || 'N/A'}<br>
                    <strong>Kiểm soát Vector:</strong> ${result.IV_Giải_pháp_Sinh_học_và_Thay_thế.Kiểm_soát_Côn_trùng_Vector || 'N/A'}
                `;
                predBiological.innerHTML = bioText.replace(/  +/g, ' ').trim();

            } else {
                // Xử lý trường hợp không tìm thấy bệnh (ví dụ: JSON thiếu ID 0)
                document.querySelector('.pred_class').innerHTML = "Không tìm thấy dữ liệu cho ID: " + class_idx;
                predClassification.innerHTML = "";
                infoContainer.style.display = "none";
            }

            // Cập nhật thanh tiến trình (như cũ)
            document.querySelector('.inner').innerHTML = `${parseFloat(prediction[class_idx]*100).toFixed(2)}%`
            progressBar.set(0);
            progressBar.animate(prediction[class_idx]-0.005); 
            pconf.style.display = 'block'
            confidence.innerHTML = Math.round(prediction[class_idx]*100)
            document.querySelector('.init_status').innerHTML = '';
        }
    );
    
}

// --- PHẦN CÒN LẠI CỦA FILE GIỮ NGUYÊN ---

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


// Logic Camera (giữ nguyên)
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


// Logic Chế độ Sáng/Tối (giữ nguyên)
modeToggle.addEventListener('click', () => {
    if (body.classList.contains('light-mode')) {
        body.classList.replace('light-mode', 'dark-mode');
        modeToggle.innerHTML = '<i class="material-icons">wb_sunny</i> Chế độ Sáng';
    } else {
        body.classList.replace('dark-mode', 'light-mode');
        modeToggle.innerHTML = '<i class="material-icons">brightness_4</i> Chế độ Tối';
    }
});
