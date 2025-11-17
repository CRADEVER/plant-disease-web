let model;
let class_indices;
let fileUpload = document.getElementById('uploadImage');
let img = document.getElementById('image');
let boxResult = document.querySelector('.box-result');
let confidence = document.querySelector('.confidence');
let pconf = document.querySelector('.box-result p');
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


let progressBar =
    new ProgressBar.Circle('#progress', {
    color: 'limegreen',
    strokeWidth: 10,
    duration: 2000, // milliseconds
    easing: 'easeInOut'
});

async function fetchData(){
    let response = await fetch('./class_indices.json');
    let data = await response.json();
    // Chuyển đổi từ key/value index -> classname sang classname -> index để dễ dàng truy cập
    let indices = {};
    for (const key in data) {
        indices[data[key]] = key;
    }
    return data;
}

// Initialize/Load model
async function initialize() {
    let status = document.querySelector('.init_status')
    status.innerHTML = 'Đang tải mô hình .... <span class="fa fa-spinner fa-spin"></span>'
    model = await tf.loadLayersModel('./tensorflowjs-model/model.json');
    status.innerHTML = 'Tải mô hình thành công <span class="fa fa-check"></span>'
    boxResult.style.display = 'block'
}

async function predict() {
    // Function for invoking prediction
    await initialize(); // Đảm bảo mô hình đã tải trước khi dự đoán
    let offset = tf.scalar(255)
    
    // Sử dụng img để lấy ảnh đã tải/chụp
    let tensorImg =   tf.browser.fromPixels(img).resizeNearestNeighbor([224,224]).toFloat().expandDims();
    let tensorImg_scaled = tensorImg.div(offset)
    
    let prediction = await model.predict(tensorImg_scaled).data();

    fetchData().then((data)=>
        {
            predicted_class = tf.argMax(prediction)
            
            class_idx = Array.from(predicted_class.dataSync())[0]
            document.querySelector('.pred_class').innerHTML = data[class_idx]
            document.querySelector('.inner').innerHTML = `${parseFloat(prediction[class_idx]*100).toFixed(2)}%`

            // Cập nhật progress bar
            progressBar.set(0); // Reset progress bar
            progressBar.animate(prediction[class_idx]-0.005); // percent

            pconf.style.display = 'block'

            confidence.innerHTML = Math.round(prediction[class_idx]*100)
            
            // Xóa trạng thái tải
            document.querySelector('.init_status').innerHTML = '';
        }
    );
    
}

// --- Xử lý tải ảnh lên ---
fileUpload.addEventListener('change', function(e){
    // Ẩn camera nếu đang mở
    stopCamera();
    cameraContainer.style.display = 'none';

    let uploadedImage = e.target.value
    if (uploadedImage){
        document.getElementById("blankFile-1").innerHTML = uploadedImage.replace("C:\\fakepath\\","")
        document.getElementById("choose-text-1").innerText = "Đổi Ảnh Đã Chọn"
        document.querySelector(".success-1").style.display = "inline-block"

        // Cập nhật màu sắc cho biểu tượng check
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
            img.style.width = "100%"; // Đảm bảo ảnh hiển thị đầy đủ
            img.style.height = "350px"; 
            predict(); // Bắt đầu dự đoán sau khi ảnh được tải
        });
    }

    else{
        img.setAttribute("src", "");
        img.style.display = "none";
    }
})


// --- Xử lý Camera ---
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
        // Vẽ frame hiện tại của video lên canvas
        canvas.width = videoStream.videoWidth;
        canvas.height = videoStream.videoHeight;
        context.drawImage(videoStream, 0, 0, canvas.width, canvas.height);
        
        // Chuyển canvas thành ảnh và gán vào thẻ <img>
        img.setAttribute('src', canvas.toDataURL('image/jpeg'));
        img.style.display = "block";
        img.style.width = "100%";
        img.style.height = "350px";

        // Tắt camera sau khi chụp
        stopCamera();
        cameraContainer.style.display = 'none';
        
        // Bắt đầu dự đoán
        predict();
    }
});

async function startCamera() {
    try {
        cameraStatus.textContent = 'Đang yêu cầu truy cập camera...';
        // Sử dụng facingMode: { exact: "environment" } cho camera sau trên mobile
        // Hoặc không chỉ định để dùng camera mặc định (trước trên laptop)
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                // Thử camera sau trên điện thoại nếu có
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
        img.style.display = 'none'; // Ẩn ảnh đã chọn trước đó
        boxResult.style.display = 'none'; // Ẩn kết quả cũ
    } catch (err) {
        // Thử lại với facingMode: 'user' nếu 'environment' thất bại
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


// --- Xử lý Chế độ Sáng/Tối ---
modeToggle.addEventListener('click', () => {
    if (body.classList.contains('light-mode')) {
        body.classList.replace('light-mode', 'dark-mode');
        modeToggle.innerHTML = '<i class="material-icons">wb_sunny</i> Chế độ Sáng';
    } else {
        body.classList.replace('dark-mode', 'light-mode');
        modeToggle.innerHTML = '<i class="material-icons">brightness_4</i> Chế độ Tối';
    }
});
