let model;
let disease_protocols_map = {}; 
let class_indices = {}; 
let currentStream;


const fileUpload = document.getElementById('uploadImage');
const img = document.getElementById('image');
const boxResult = document.getElementById('boxResult');
const predClassSpan = document.querySelector('.pred_class');
const confidenceSpan = document.querySelector('.confidence');
const resultContainer = document.getElementById('resultContainer'); 
const mainStatus = document.getElementById('mainStatus'); 
const loadingPredictionBar = document.getElementById('loadingPredictionBar'); 


const cameraToggle = document.getElementById('cameraToggle');
const cameraContainer = document.getElementById('cameraContainer');
const videoStream = document.getElementById('videoStream');
const captureButton = document.getElementById('captureButton');
const stopButton = document.getElementById('stopButton');
const cameraStatus = document.getElementById('cameraStatus');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');


const modeToggle = document.getElementById('modeToggle');
const body = document.body;

const progressBar = new ProgressBar.Circle('#progress', {
    color: '#00a896', 
    strokeWidth: 10,
    duration: 1000,
    easing: 'easeInOut',
    trailColor: '#e0e0e0', 
    trailWidth: 4,
    svgStyle: null
});



async function fetchData(){
    try {
        let response = await fetch('./class_indices.json');
        
        if (!response.ok) {
            throw new Error(`HTTP Error! Status: ${response.status}. Hãy đảm bảo file class_indices.json nằm cùng cấp với index.html.`);
        }
        
        let data = await response.json();
        
        let protocolMap = {};
        let indicesMap = {};
        
        // **Đã sửa:** Sử dụng tên key đã thống nhất (Phac_do_Quan_Ly_Tong_Hop_Chi_tiet)
        const protocolsArray = data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet; 

        if (Array.isArray(protocolsArray)) {
            protocolsArray.forEach(item => {
                // **Đã sửa:** Truy cập key bằng cú pháp chuỗi
                protocolMap[item['Mã id']] = item; 
                indicesMap[item['Mã id']] = item['Tên bệnh']; 
            });
        } else {
             throw new Error("Cấu trúc JSON không hợp lệ: 'Phac_do_Quan_Ly_Tong_Hop_Chi_tiet' không phải là mảng hoặc không tồn tại.");
        }

        disease_protocols_map = protocolMap;
        class_indices = indicesMap; 
        
        console.log("DEBUG: class_indices.json đã tải thành công và được ánh xạ.", { class_indices, disease_protocols_map });

    } catch (error) {
        console.error("Lỗi khi tải class_indices.json:", error);
        mainStatus.className = 'status error';
        mainStatus.innerHTML = `<i class="material-icons">error_outline</i> Lỗi: Không thể tải phác đồ quản lý bệnh. Chi tiết: ${error.message}`;
        return null;
    }
}


async function initialize() {
    mainStatus.className = 'status loading';
    mainStatus.innerHTML = '<i class="material-icons loading-icon">cached</i> Đang tải mô hình và dữ liệu quản lý...';
    

    await fetchData();

 
    try {
        const modelUrl = './tensorflowjs-model/model.json'; 
        // Đây là bước tải mô hình, giả định đã có tf.js được include
        model = await tf.loadLayersModel(modelUrl); 

        mainStatus.className = 'status success';
        mainStatus.innerHTML = '<i class="material-icons">check_circle_outline</i> Hệ thống đã sẵn sàng. Hãy chọn ảnh hoặc dùng Camera.';
        
    } catch (error) {
        console.error("Lỗi khi tải mô hình TensorFlow.js:", error);
        mainStatus.className = 'status error';
        mainStatus.innerHTML = '<i class="material-icons">error_outline</i> Lỗi: Không thể tải mô hình dự đoán. Đảm bảo thư mục <b>tensorflowjs-model</b> chứa <b>model.json</b> và các file <b>.bin</b>.';
    }
}



function displayDiseaseDetails(protocol) {
    resultContainer.style.display = 'block';
    
    // **Đã sửa:** Truy cập key bằng cú pháp chuỗi
    const chemicalProtocol = protocol['Iii. chiến lược kiểm soát hóa học'];
    const phacDoGiaiDoan = chemicalProtocol?.['Phác đồ giai đoạn cây'] || [];

   
    // **Đã sửa:** Truy cập key bằng cú pháp chuỗi
    const intensiveCultivationProtocol = protocol['Ii. biện pháp canh tác chuyên sâu'];
    let sectionIICulturalContent = '';

    if (intensiveCultivationProtocol) {
        sectionIICulturalContent = `
            <ul>
                <li><b>Quản lý Tàn dư & Đất:</b> ${intensiveCultivationProtocol['Quản lý tàn dư đất'] || 'Đang cập nhật...'}</li>
                <li><b>Quản lý Nước Tưới:</b> ${intensiveCultivationProtocol['Quản lý nước tưới'] || 'Đang cập nhật...'}</li>
                <li><b>Mật độ & Thông thoáng:</b> ${intensiveCultivationProtocol['Mật độ thông thoáng'] || 'Đang cập nhật...'}</li>
                <li><b>Quản lý Dinh dưỡng Vi lượng:</b> ${intensiveCultivationProtocol['Quản lý dinh dưỡng vi lượng'] || 'Đang cập nhật...'}</li>
            </ul>
        `;
    } else {
         sectionIICulturalContent = '<p>Đang cập nhật chiến lược Canh tác chuyên sâu...</p>';
    }

 
    // **Đã sửa:** Truy cập key bằng cú pháp chuỗi
    const bioProtocol = protocol['Iv giải pháp sinh học và thay thế'];
    let sectionIVBioContent = '';
    
    if (bioProtocol) {
        sectionIVBioContent = `
            <p><b>Chất Đối kháng VSV:</b> ${bioProtocol['Chất đối kháng vsv'] || 'Đang cập nhật...'}</p>
            <p><b>Cảm ứng Kháng Bệnh (SAR):</b> ${bioProtocol['Cảm ứng kháng bệnh sar'] || 'Không có.'}</p>
            <p><b>Kiểm soát Côn trùng Vector:</b> ${bioProtocol['Kiểm soát côn trùng vector'] || 'Đang cập nhật...'}</p>
            <p><b>Quản lý Kháng thuốc (IRM):</b> Không có thông tin IRM trong dữ liệu.</p>
        `;
    } else {
        sectionIVBioContent = '<p>Đang cập nhật giải pháp sinh học và thay thế...</p>';
    }
    
    
    let html = `
        <div class="protocol-header">
            <h3>${protocol['Tên bệnh'] || 'Chưa xác định'}</h3>
            <p class="classification">Phân loại: <b>${protocol['Phân loại'] || 'Đang cập nhật'}</b></p>
        </div>
        <hr>
        
        <div class="protocol-sections">
            <details class="protocol-detail-section" open>
                <summary>
                    <i class="material-icons">science</i>
                    I. Tác nhân, Chu kỳ và Điều kiện (Cơ sở)
                </summary>
                <div class="detail-content">
                    <p><b>Tác nhân:</b> ${protocol['I. tác nhân chu kỳ và điều kiện']?.['Tác nhân sinh học'] || 'Đang cập nhật...'}</p>
                    <p><b>Cơ chế lây lan:</b> ${protocol['I. tác nhân chu kỳ và điều kiện']?.['Cơ chế lây lan'] || 'Đang cập nhật...'}</p>
                    <p><b>Nhiệt độ tối ưu:</b> ${protocol['I. tác nhân chu kỳ và điều kiện']?.['Nhiệt độ thời điểm tối ưu'] || 'Đang cập nhật...'}</p>
                    <p><b>Dấu hiệu:</b> ${protocol['I. tác nhân chu kỳ và điều kiện']?.['Dấu hiệu chẩn đoán chuyên sâu'] || 'Đang cập nhật...'}</p>
                </div>
            </details>

            <details class="protocol-detail-section">
                <summary>
                    <i class="material-icons">agriculture</i>
                    II. Biện pháp Canh tác Chuyên sâu (Phòng ngừa)
                </summary>
                <div class="detail-content">
                    ${sectionIICulturalContent}
                </div>
            </details>

            <details class="protocol-detail-section">
                <summary>
                    <i class="material-icons">local_florist</i>
                    III. Chiến lược Kiểm soát Hóa học (Thuốc)
                </summary>
                <div class="detail-content">
                    <p><b>Nguyên tắc FRAC/IRAC:</b> ${chemicalProtocol?.['Nguyên tắc frac irac'] || 'Đang cập nhật...'}</p>
                    
                    <h4>Phác đồ theo Giai đoạn:</h4>
                    ${phacDoGiaiDoan.length > 0 ? phacDoGiaiDoan.map(step => `
                        <div class="stage-step">
                            <p><b>Giai đoạn:</b> ${step['Giai đoạn'] || 'N/A'}</p>
                            <p><b>Hoạt chất đề xuất:</b> <span>${step['Hoạt chất đề xuất'] || 'N/A'}</span> (Nhóm: ${step['Nhóm frac irac'] || 'N/A'})</p>
                            <p><b>Lưu ý:</b> ${step['Lưu ý ứng dụng'] || 'N/A'}</p>
                        </div>
                    `).join('') : '<p>Đang cập nhật phác đồ giai đoạn hóa học...</p>'}
                    <p><b>Thuốc Trừ Tận gốc Eradicant:</b> ${chemicalProtocol?.['Thuốc trừ tận gốc eradicant'] || 'Không sử dụng.'}</p>
                </div>
            </details>
            
            <details class="protocol-detail-section">
                <summary>
                    <i class="material-icons">hive</i>
                    IV. Giải pháp Sinh học và Thay thế
                </summary>
                <div class="detail-content">
                    ${sectionIVBioContent}
                </div>
            </details>
        </div>
    `;

    resultContainer.innerHTML = html;
}


async function predict(imageElement) {
    if (!model) {
        mainStatus.className = 'status error';
        mainStatus.innerHTML = '<i class="material-icons">error_outline</i> Mô hình chưa được tải. Vui lòng kiểm tra console.';
        return;
    }
    
 
    resultContainer.style.display = 'none';
    boxResult.style.display = 'flex'; 
    loadingPredictionBar.style.display = 'flex'; 
    progressBar.set(0);
    confidenceSpan.textContent = 0;
    predClassSpan.textContent = 'Đang phân tích...';
    
    let predicted_index, confidence_score;
    
    try {
 
        const tensor = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([224, 224]) 
            .toFloat()
            .div(tf.scalar(255.0)) 
            .expandDims(); 

        
        const predictions = model.predict(tensor);
        const predictionArray = await predictions.data();
        
     
        const highestPrediction = Math.max(...predictionArray);
        
        predicted_index = predictionArray.indexOf(highestPrediction).toString(); 
        confidence_score = Math.floor(highestPrediction * 100);

        tensor.dispose(); 
        predictions.dispose();

    } catch (e) {
        console.error("Lỗi khi chạy dự đoán TensorFlow.js:", e);
        loadingPredictionBar.style.display = 'none';
        predClassSpan.textContent = 'Lỗi Phân Tích!';
        confidenceSpan.textContent = 0;
        resultContainer.style.display = 'block';
        resultContainer.innerHTML = `<div class="protocol-header error">
            <i class="material-icons">warning</i> 
            Lỗi trong quá trình xử lý ảnh và dự đoán. Vui lòng kiểm tra console để biết chi tiết lỗi TensorFlow.
        </div>`;
        return;
    }
    
    
    loadingPredictionBar.style.display = 'none';

    let normalizedConfidence = confidence_score / 100;
    progressBar.animate(normalizedConfidence, () => {
        confidenceSpan.textContent = confidence_score;
    });

    const diseaseName = class_indices[predicted_index] || "Không xác định (Mã: " + predicted_index + ")";
    predClassSpan.textContent = diseaseName;
    
    console.log(`DEBUG: Index dự đoán: ${predicted_index}. Confidence: ${confidence_score}%. Bệnh: ${diseaseName}`); 
    
    const protocol = disease_protocols_map[predicted_index];

    if (protocol) {
        console.log("DEBUG: Đã tìm thấy Phác đồ chi tiết. Bắt đầu hiển thị."); 
        displayDiseaseDetails(protocol);
    } else {
        console.error("DEBUG: KHÔNG tìm thấy Phác đồ cho Mã_ID:", predicted_index); 
        resultContainer.innerHTML = `<div class="protocol-header error">
            <i class="material-icons">warning</i> 
            Không tìm thấy phác đồ quản lý chi tiết cho bệnh <b>${diseaseName}</b>. (Mã: ${predicted_index}).
        </div>`;
        resultContainer.style.display = 'block';
    }
}



async function startCamera() {
    boxResult.style.display = 'none';
    resultContainer.style.display = 'none';
    img.style.display = 'none';
    document.querySelector('.image-placeholder').style.display = 'none';
    
    cameraContainer.style.display = 'block';

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        try {
            // Yêu cầu camera sau (environment) trên thiết bị di động
            currentStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            videoStream.srcObject = currentStream;
            videoStream.play();
            cameraStatus.textContent = 'Camera đã sẵn sàng. Hãy chụp ảnh.';
            captureButton.disabled = false;
            videoStream.style.display = 'block';
            captureButton.style.display = 'flex';
            stopButton.style.display = 'flex';
        } catch (error) {
            cameraStatus.textContent = `Lỗi truy cập camera: ${error.name}. Vui lòng đảm bảo camera được phép sử dụng.`;
            captureButton.disabled = true;
            videoStream.style.display = 'none';
            captureButton.style.display = 'none';
            stopButton.style.display = 'none';
            cameraContainer.style.display = 'block';
        }
    } else {
        cameraStatus.textContent = 'Trình duyệt không hỗ trợ Media Devices API.';
        cameraContainer.style.display = 'block';
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
    cameraContainer.style.display = 'none';
    captureButton.style.display = 'none';
    stopButton.style.display = 'none';
    document.querySelector('.image-placeholder').style.display = 'block';
}


fileUpload.addEventListener('change', function () {
    const file = this.files[0];
    if (file) {
        stopCamera();
        const reader = new FileReader();
        reader.onload = function (e) {
            img.src = e.target.result;
            img.style.display = 'block'; 
            document.querySelector('.image-placeholder').style.display = 'none';
            img.onload = () => {
                predict(img);
            };
        };
        reader.readAsDataURL(file);
    }
});



cameraToggle.addEventListener('click', function() {
    if (!currentStream) {
        startCamera();
    } else {
        stopCamera();
    }
});

captureButton.addEventListener('click', () => {
    resultContainer.style.display = 'none'; 
    
    // Tải ảnh từ video stream lên canvas
    canvas.width = 224;
    canvas.height = 224;
    context.drawImage(videoStream, 0, 0, canvas.width, canvas.height); 
    
    const dataUrl = canvas.toDataURL('image/png');
    img.src = dataUrl;
    img.style.display = 'block'; 
    
    // Ẩn camera UI
    videoStream.style.display = 'none';
    captureButton.style.display = 'none';
    stopButton.style.display = 'none';
    cameraContainer.style.display = 'none';
    
    cameraStatus.textContent = 'Ảnh đã được chụp. Đang phân tích...';
    
    predict(img);
});


stopButton.addEventListener('click', stopCamera);


modeToggle.addEventListener('click', () => {
    const isLightMode = body.classList.contains('light-mode');
    if (isLightMode) {
        body.classList.replace('light-mode', 'dark-mode');
        modeToggle.innerHTML = '<i class="material-icons">wb_sunny</i> Chế độ Sáng';
        progressBar.options.trailColor = '#333333';
    } else {
        body.classList.replace('dark-mode', 'light-mode');
        modeToggle.innerHTML = '<i class="material-icons">brightness_4</i> Chế độ Tối';
        progressBar.options.trailColor = '#e0e0e0';
    }
    progressBar.set(progressBar.value()); 
});



document.addEventListener('DOMContentLoaded', initialize);
