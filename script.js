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

// NEW: Global variable for detailed data lookup
let class_indices_map = {}; 
let detailedResultContainer = document.getElementById('detailedResultContainer');


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
   
    let indices = {}; // Map: Index (0, 1, 2...) -> Class Name
    
    // Populate indices map and the new class_indices_map for detailed lookup
    data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet.forEach(item => {
        // The ML model output is an index, which corresponds to Mã_ID (e.g., "0", "1", "2")
        class_indices_map[item.Mã_ID] = item;
        indices[item.Mã_ID] = item.Tên_Bệnh; 
    });

    class_indices = indices; // Update the global variable
    
    // The full array of detailed data
    return data.Phac_do_Quan_Ly_Tong_Hop_Chi_tiet; 
}


async function initialize() {
    let status = document.querySelector('.init_status');
    if (status) { // Check for status element to prevent TypeError
        status.innerHTML = 'Đang tải dữ liệu mô hình...';
    }
    
    await fetchData(); 
    
    if (status) { // Check for status element
        status.innerHTML = 'Đang tải mô hình học sâu...';
    }
    try {
        // Assuming the model is in a 'model' directory
        model = await tf.loadGraphModel('./model/model.json'); 
        if (status) {
            status.innerHTML = 'Hệ thống đã sẵn sàng!';
            status.style.backgroundColor = '#28a745';
        }
        
    } catch (error) {
        if (status) {
            status.innerHTML = 'Lỗi tải mô hình. Vui lòng kiểm tra file model.json.';
            status.style.backgroundColor = '#dc3545';
        }
        console.error("Error loading model:", error);
    }
}

function processImage(file) {
    if (file) {
        let reader = new FileReader();
        reader.onload = function(e) {
            img.src = e.target.result;
            img.style.display = 'block';
            videoStream.style.display = 'none';
            captureButton.style.display = 'none';
            stopButton.style.display = 'none';
            if (boxResult) boxResult.style.display = 'none'; // Null check
            if (pconf) pconf.style.display = 'none'; // Null check
            if (detailedResultContainer) detailedResultContainer.style.display = 'none'; // Null check
            
            // Wait for image to load before prediction
            img.onload = () => {
                predict();
            }
        };
        reader.readAsDataURL(file);
    }
}


async function predict() {
    // FIX: Check if essential DOM elements exist to prevent TypeError (Cannot set properties of null)
    if (!boxResult || !pconf || !confidence || !model) {
        console.error("Lỗi: Mô hình hoặc các phần tử DOM cần thiết chưa được tải/tìm thấy.");
        return; 
    }
    
    // Hide previous detailed result 
    if (detailedResultContainer) {
        detailedResultContainer.style.display = 'none';
    }
    
    // Preprocessing the image for the model
    let tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224]) // Resize to model input size
        .toFloat()
        .div(tf.scalar(255.0)) // Normalize
        .expandDims(); // Add batch dimension

    // Show loading indicator
    boxResult.style.display = 'block';
    pconf.style.display = 'block';
    document.querySelector('.pred_class').textContent = 'Đang phân tích...';
    confidence.textContent = '...';
    progressBar.set(0);

    // Run prediction
    let prediction = await model.predict(tensor).data();
    tensor.dispose(); // Clean up tensor memory

    // Get the result
    let max_index = prediction.indexOf(Math.max(...prediction));
    let predicted_class_name = class_indices[max_index];
    let conf = Math.floor(Math.max(...prediction) * 100);

    // Update prediction summary
    pconf.style.display = 'block';
    confidence.textContent = conf;
    document.querySelector('.pred_class').textContent = predicted_class_name;
    boxResult.style.display = 'block';

    progressBar.animate(conf / 100);

    // NEW: Display detailed result
    const resultData = class_indices_map[max_index.toString()]; 
    if (resultData) {
        displayDetailedResult(resultData);
    } else {
        if (detailedResultContainer) {
            detailedResultContainer.style.display = 'block';
            detailedResultContainer.innerHTML = '<h3>Không tìm thấy thông tin chi tiết cho loại bệnh này.</h3>';
        }
    }
}


// NEW: Recursive function to format any value (string, object, array)
function formatValue(key, value) {
    let html = '';
    let displayKey = key.replace(/_/g, ' ');

    if (value === null || value === "" || (Array.isArray(value) && value.length === 0)) {
        // Skip empty or null values
        return '';
    }

    if (Array.isArray(value)) {
        // Handle Arrays (e.g., Phác_đồ_Giai_đoạn_Cây)
        html += `<h4>${displayKey}:</h4><ul>`;
        value.forEach((item, index) => {
            html += `<li><strong>Mục ${index + 1}:</strong>`;
            // Recursively process array object items
            html += formatObject(item, true); // true indicates it's a list item sub-object
            html += `</li>`;
        });
        html += `</ul>`;

    } else if (typeof value === 'object' && value !== null) {
        // Handle Nested Objects (e.g., Dấu_hiệu_Chẩn_đoán_Chuyên_sâu)
        html += `<h4>${displayKey}:</h4>`;
        html += formatObject(value, false);

    } else {
        // Handle Simple Key-Value Pairs (String/Number)
        // If it's part of an array item, return simple paragraph with strong key
        if (key === 'Giai_đoạn' || key === 'Hoạt_chất_Đề_xuất' || key === 'Nhóm_FRAC_IRAC' || key === 'Lưu_ý_Ứng_dụng') {
             return `<p><strong>${displayKey}:</strong> ${value}</p>`;
        }
        
        // For regular paragraphs
        html += `<h4>${displayKey}:</h4><p>${value}</p>`;
    }

    return html;
}

// NEW: Helper for formatting objects, called by formatValue
function formatObject(obj, isListItem = false) {
    let html = isListItem ? '<ul>' : ''; // Start a sub-list if it's within a list item

    for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
            // If it's a list item sub-object, render keys as bold strings inside LIs
            if (isListItem) {
                // Call formatValue to handle potential nesting within the list item object (e.g., if there were sub-sub-objects)
                // Since the array items are simple key-value pairs, we can just render them directly:
                let displayKey = key.replace(/_/g, ' ');
                html += `<li><strong>${displayKey}:</strong> ${obj[key]}</li>`;
            } else {
                // Otherwise, call formatValue recursively for the next level
                html += formatValue(key, obj[key]);
            }
        }
    }
    
    if (isListItem) {
        html += '</ul>';
    }
    return html;
}


// NEW: Function to display the full, monolithic analysis content
function displayDetailedResult(result) {
    // FIX: Check if the element is null to avoid the TypeError
    if (!detailedResultContainer) {
        console.error("Lỗi: Không tìm thấy phần tử #detailedResultContainer.");
        return;
    }

    let htmlContent = `
        <h2>${result.Tên_Bệnh} (ID: ${result.Mã_ID})</h2>
        <p><strong>Phân loại tổng quát:</strong> ${result.Phân_loại}</p>
        <hr/>
    `;

    // Define the section order based on JSON structure
    const sectionOrder = [
        "I_Tác_nhân_Chu_kỳ_và_Điều_kiện",
        "II_Biện_pháp_Canh_tác_Chuyên_sâu",
        "III_Chiến_lược_Kiểm_soát_Hóa_học",
        "IV_Giải_pháp_Sinh_học_và_Thay_thế"
    ];

    sectionOrder.forEach(sectionKey => {
        if (result[sectionKey]) {
            let sectionTitle = sectionKey.replace(/_/g, ' ');
            // The main section is an object, so we call formatObject
            htmlContent += `<h3>${sectionTitle}</h3>`;
            htmlContent += formatObject(result[sectionKey], false);
        }
    });

    detailedResultContainer.innerHTML = htmlContent;
    detailedResultContainer.style.display = 'block';
}


fileUpload.addEventListener('change', (e) => {
    processImage(e.target.files[0]);
});

captureButton.addEventListener('click', () => {
    // Take a snapshot from the video stream
    canvas.width = videoStream.videoWidth;
    canvas.height = videoStream.videoHeight;
    context.drawImage(videoStream, 0, 0, canvas.width, canvas.height);

    // Convert canvas to base64 image data
    let dataUrl = canvas.toDataURL('image/jpeg');

    // Display the captured image
    img.src = dataUrl;
    img.style.display = 'block';
    videoStream.style.display = 'none';
    captureButton.style.display = 'none';
    stopCamera(); // Stop the camera after capture

    // Process the captured image
    img.onload = () => {
        predict();
    }
});


cameraToggle.addEventListener('click', () => {
    if (cameraContainer.style.display === 'block') {
        stopCamera();
        cameraContainer.style.display = 'none';
    } else {
        startCamera();
        cameraContainer.style.display = 'block';
        img.style.display = 'none';
        if (boxResult) boxResult.style.display = 'none'; // Null check
        if (detailedResultContainer) detailedResultContainer.style.display = 'none'; // Null check
    }
});

stopButton.addEventListener('click', () => {
    stopCamera();
    cameraContainer.style.display = 'none';
    if (detailedResultContainer) detailedResultContainer.style.display = 'none'; // Null check
});


async function startCamera() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
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
            if (boxResult) boxResult.style.display = 'none'; // Null check
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

// Start initialization when the page loads
window.onload = initialize;
