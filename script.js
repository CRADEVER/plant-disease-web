let model;
let class_indices;
let fileUpload = document.getElementById('uploadImage');
let img = document.getElementById('image');
let boxResult = document.querySelector('.box-result');
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

// ==================================================
// HÀM ĐIỀU KHIỂN TAB MỚI
// ==================================================
function openTab(evt, tabName) {
    let i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tab-link");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    if (evt) {
        evt.currentTarget.className += " active";
    } else {
        // Mặc định chọn tab đầu tiên
        document.querySelector(".tab-link").className += " active";
    }
}

// ==================================================
// HÀM DỌN DẸP KẾT QUẢ CŨ
// ==================================================
function clearPrediction() {
    document.querySelector('.pred_class').innerHTML = '';
    document.querySelector('.confidence').innerHTML = '';
    pconf.style.display = 'none';
    progressBar.set(0);
    document.querySelector('.inner').innerHTML = '';

    // Xóa nội dung các tab thông tin
    document.getElementById('info-classification').innerHTML = '';
    document.getElementById('info-symptoms').innerHTML = '';
    document.getElementById('info-cult-residue').innerHTML = '';
    document.getElementById('info-cult-water').innerHTML = '';
    document.getElementById('info-cult-density').innerHTML = '';
    document.getElementById('info-cult-nutrition').innerHTML = '';
    document.getElementById('info-chem-principle').innerHTML = '';
    document.getElementById('info-chem-plan').getElementsByTagName('tbody')[0].innerHTML = '';
    document.getElementById('info-bio-antagonist').innerHTML = '';
    document.getElementById('info-bio-sar').innerHTML = '';
    document.getElementById('info-bio-vector').innerHTML = '';

    // Ẩn container thông tin
    let infoContainer = document.querySelector('.disease-info-container');
    if (infoContainer) {
        infoContainer.style.display = 'none';
    }

    // Reset về tab đầu tiên
    openTab(null, 'tab-overview');
}


// ==================================================
// FETCHDATA ĐÃ SỬA
// ==================================================
async function fetchData() {
    let response = await fetch('./class_indices.json');
    let data = await response.json();
    // Trả về mảng các đối tượng bệnh
    return data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet;
}

// ==================================================
// INITIALIZE ĐÃ SỬA (Chỉ tải mô hình, không làm gì khác)
// ==================================================
async function initialize() {
    let status = document.querySelector('.init_status');
    status.innerHTML = 'Đang tải mô hình .... <span class="fa fa-spinner fa-spin"></span>';
    try {
        model = await tf.loadLayersModel('./tensorflowjs-model/model.json');
        status.innerHTML = 'Mô hình đã sẵn sàng! <span class="fa fa-check"></span>';
    } catch (e) {
        status.innerHTML = 'Lỗi tải mô hình. Vui lòng tải lại trang.';
        console.error(e);
    }
}

// ==================================================
// HÀM PREDICT ĐÃ CẢI TIẾN
// ==================================================
async function predict() {
    // 1. Dọn dẹp kết quả cũ và hiển thị trạng thái
    clearPrediction();
    boxResult.style.display = 'block';
    let status = document.querySelector('.init_status');

    // 2. Kiểm tra và tải mô hình nếu cần
    if (!model) {
        status.innerHTML = 'Đang tải mô hình .... <span class="fa fa-spinner fa-spin"></span>';
        await initialize();
    }
    
    status.innerHTML = 'Đang phân tích hình ảnh... <span class="fa fa-spinner fa-spin"></span>';

    // 3. Xử lý ảnh và dự đoán
    let offset = tf.scalar(255);
    let tensorImg = tf.browser.fromPixels(img).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
    let tensorImg_scaled = tensorImg.div(offset);
    
    let prediction = await model.predict(tensorImg_scaled).data();

    // 4. Lấy dữ liệu JSON và hiển thị
    fetchData().then((diseaseArray) => {
        let predicted_class = tf.argMax(prediction);
        let class_idx = Array.from(predicted_class.dataSync())[0];
        
        // Lấy đúng đối tượng bệnh từ mảng
        let diseaseInfo = diseaseArray[class_idx];

        if (!diseaseInfo) {
            status.innerHTML = 'Lỗi: Không tìm thấy thông tin bệnh.';
            return;
        }

        // --- Điền thông tin cơ bản ---
        document.querySelector('.pred_class').innerHTML = diseaseInfo.Tên_Bệnh;
        let confidenceValue = parseFloat(prediction[class_idx] * 100).toFixed(2);
        document.querySelector('.inner').innerHTML = `${confidenceValue}%`;
        
        progressBar.set(0);
        progressBar.animate(prediction[class_idx] - 0.005);
        pconf.style.display = 'block';
        document.querySelector('.confidence').innerHTML = Math.round(prediction[class_idx] * 100);
        
        // --- Điền thông tin chi tiết vào các TAB ---
        
        // Tab 1: Tổng quan & Triệu chứng
        document.getElementById('info-classification').innerHTML = diseaseInfo.Phân_loại || 'Không có thông tin.';
        let symptoms = diseaseInfo.I_Tác_nhân_Chu_kỳ_và_Điều_kiện.Dấu_hiệu_Chẩn_đoán_Chuyên_sâu;
        let symptomsHtml = '';
        if (typeof symptoms === 'object' && symptoms !== null) {
            symptomsHtml = '<ul>';
            for (const key in symptoms) {
                if (symptoms.hasOwnProperty(key)) {
                    symptomsHtml += `<li><strong>${key.replace(/_/g, ' ')}:</strong> ${symptoms[key]}</li>`;
                }
            }
            symptomsHtml += '</ul>';
        } else {
            symptomsHtml = `<p>${symptoms || 'Không có thông tin.'}</p>`;
        }
        document.getElementById('info-symptoms').innerHTML = symptomsHtml;

        // Tab 2: Canh tác
        let cultivation = diseaseInfo.II_Biện_pháp_Canh_tác_Chuyên_sâu;
        document.getElementById('info-cult-residue').innerHTML = cultivation.Quản_lý_Tàn_dư_Đất || 'Không có thông tin.';
        document.getElementById('info-cult-water').innerHTML = cultivation.Quản_lý_Nước_Tưới || 'Không có thông tin.';
        document.getElementById('info-cult-density').innerHTML = cultivation.Mật_độ_Thông_thoáng || 'Không có thông tin.';
        document.getElementById('info-cult-nutrition').innerHTML = cultivation.Quản_lý_Dinh_dưỡng_Vi_lượng || 'Không có thông tin.';

        // Tab 3: Hóa học
        let chemical = diseaseInfo.III_Chiến_lược_Kiểm_soát_Hóa_học;
        document.getElementById('info-chem-principle').innerHTML = chemical.Nguyên_tắc_FRAC_IRAC || 'Không có thông tin.';
        let chemPlanTable = document.getElementById('info-chem-plan').getElementsByTagName('tbody')[0];
        chemPlanTable.innerHTML = ''; // Xóa các hàng cũ
        if (chemical.Phác_đồ_Giai_đoạn_Cây && chemical.Phác_đồ_Giai_đoạn_Cây.length > 0) {
            chemical.Phác_đồ_Giai_đoạn_Cây.forEach(plan => {
                let row = chemPlanTable.insertRow();
                row.insertCell(0).innerHTML = plan.Giai_đoạn;
                row.insertCell(1).innerHTML = plan.Hoạt_chất_Đề_xuất;
                row.insertCell(2).innerHTML = plan.Nhóm_FRAC_IRAC;
                row.insertCell(3).innerHTML = plan.Lưu_ý_Ứng_dụng;
            });
        } else {
            let row = chemPlanTable.insertRow();
            row.insertCell(0).innerHTML = 'Không có phác đồ.';
            row.insertCell(1).innerHTML = '-';
            row.insertCell(2).innerHTML = '-';
            row.insertCell(3).innerHTML = '-';
        }

        // Tab 4: Sinh học
        let biological = diseaseInfo.IV_Giải_pháp_Sinh_học_và_Thay_thế;
        document.getElementById('info-bio-antagonist').innerHTML = biological.Chất_Đối_kháng_VSV || 'Không có thông tin.';
        document.getElementById('info-bio-sar').innerHTML = biological.Cảm_ứng_Kháng_Bệnh_SAR || 'Không có thông tin.';
        document.getElementById('info-bio-vector').innerHTML = biological.Kiểm_soát_Côn_trùng_Vector || 'Không có thông tin.';

        // Hiển thị container thông tin
        document.querySelector('.disease-info-container').style.display = 'block';
        status.innerHTML = 'Phân tích hoàn tất! <span class="fa fa-check"></span>';
    
    }).catch(err => {
        console.error("Lỗi khi fetch hoặc xử lý dữ liệu JSON:", err);
        status.innerHTML = 'Lỗi khi tải thông tin bệnh.';
    });
}


fileUpload.addEventListener('change', function(e){
    // Dừng camera nếu đang chạy
    stopCamera();
    cameraContainer.style.display = 'none';

    let uploadedImage = e.target.value;
    if (uploadedImage){
        document.getElementById("blankFile-1").innerHTML = uploadedImage.replace("C:\\fakepath\\","");
        document.getElementById("choose-text-1").innerText = "Đổi Ảnh Đã Chọn";
        document.querySelector(".success-1").style.display = "inline-block";
        document.querySelector(".success-1 i").style.border = "1px solid limegreen";
        document.querySelector(".success-1 i").style.color = "limegreen";
    }
    
    let file = this.files[0];
    if (file){
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.addEventListener("load", function(){
            img.style.display = "block";
            img.setAttribute('src', this.result);
            img.style.width = "100%";
            img.style.height = "350px"; 
            // Gọi hàm predict
            predict(); 
        });
    } else {
        img.setAttribute("src", "");
        img.style.display = "none";
        // Dọn dẹp kết quả nếu không chọn file
        clearPrediction();
        boxResult.style.display = 'none';
    }
});



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
        
        // Gọi hàm predict
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
                facingMode: 'environment' // Ưu tiên camera sau
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
        clearPrediction(); // Xóa kết quả cũ
        boxResult.style.display = 'none'; // Ẩn box kết quả
    } catch (err) {
        // Thử camera trước nếu camera sau thất bại
        try {
             const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user' // Camera trước
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
            clearPrediction();
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



modeToggle.addEventListener('click', () => {
    if (body.classList.contains('light-mode')) {
        body.classList.replace('light-mode', 'dark-mode');
        modeToggle.innerHTML = '<i class="material-icons">wb_sunny</i> Chế độ Sáng';
    } else {
        body.classList.replace('dark-mode', 'light-mode');
        modeToggle.innerHTML = '<i class="material-icons">brightness_4</i> Chế độ Tối';
    }
});

// ==================================================
// TẢI MÔ HÌNH KHI TRANG ĐƯỢC MỞ
// ==================================================
initialize();
