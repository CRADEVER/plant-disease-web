let model;
let class_indices;
let fileUpload = document.getElementById('uploadImage');
let img = document.getElementById('image');
let boxResult = document.querySelector('.box-result');
let confidence = document.querySelector('.confidence');
let pconf = document.querySelector('.box-result p');

// Biến và element mới cho Camera
const activateCameraBtn = document.getElementById('activate-camera-btn');
const captureImageBtn = document.getElementById('capture-image-btn');
const videoStream = document.getElementById('camera-stream');
const cameraCanvas = document.getElementById('camera-canvas');
let currentStream = null; // Biến để lưu trữ luồng media

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
    data = JSON.stringify(data);
    data = JSON.parse(data);
    return data;
}

// Initialize/Load model
async function initialize() {
    let status = document.querySelector('.init_status')
    status.innerHTML = 'Đang tải Mô hình .... <span class="fa fa-spinner fa-spin"></span>'
    model = await tf.loadLayersModel('./tensorflowjs-model/model.json');
    status.innerHTML = 'Mô hình đã tải thành công <span class="fa fa-check"></span>'
}

async function predict() {
    // Function for invoking prediction
    let img = document.getElementById('image')
    let offset = tf.scalar(255)
    // resizeNearestNeighbor([224,224]) là bước quan trọng để đảm bảo đầu vào chuẩn
    let tensorImg =   tf.browser.fromPixels(img).resizeNearestNeighbor([224,224]).toFloat().expandDims();
    let tensorImg_scaled = tensorImg.div(offset)
    prediction = await model.predict(tensorImg_scaled).data();
    
    fetchData().then((data)=> 
        {
            predicted_class = tf.argMax(prediction)
            
            class_idx = Array.from(predicted_class.dataSync())[0]
            document.querySelector('.pred_class').innerHTML = data[class_idx]
            document.querySelector('.inner').innerHTML = `${parseFloat(prediction[class_idx]*100).toFixed(2)}% SURE`
            
            progressBar.animate(prediction[class_idx]-0.005); // percent

            pconf.style.display = 'block'

            confidence.innerHTML = Math.round(prediction[class_idx]*100)

            // Ẩn trạng thái tải sau khi hoàn thành
            document.querySelector('.init_status').style.display = 'none';
        }
    );
    
}

// Hàm Dừng Camera
function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    videoStream.style.display = 'none';
    activateCameraBtn.style.display = 'block';
    captureImageBtn.style.display = 'none';
    document.querySelector('.init_status').style.display = 'none'; // Ẩn trạng thái
}

// Hàm Khởi tạo Camera
async function activateCamera() {
    // 1. Dừng camera nếu đang chạy
    if (currentStream) {
        stopCamera();
    }
    
    // Ẩn box kết quả và ảnh mẫu
    boxResult.style.display = 'none';
    img.style.display = 'none';

    try {
        // Cấu hình linh hoạt: ưu tiên camera sau (environment) và kích thước lý tưởng
        const constraints = {
            video: {
                width: { ideal: 1280 }, 
                height: { ideal: 720 },
                facingMode: { ideal: 'environment' } 
            }
        };

        // Lấy luồng media
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        videoStream.srcObject = currentStream;
        videoStream.style.display = 'block'; // Hiển thị video
        
        activateCameraBtn.style.display = 'none';
        captureImageBtn.style.display = 'block';

        // Bắt đầu phát video
        videoStream.play();

    } catch (err) {
        console.error("Error accessing camera: ", err);
        alert("Không thể truy cập camera. Vui lòng cấp quyền.");
        activateCameraBtn.style.display = 'block';
        captureImageBtn.style.display = 'none';
    }
}

// Hàm Chụp ảnh và xử lý Cropping
function captureImage() {
    if (!videoStream.srcObject) return;

    // Lấy kích thước thực của video
    const videoWidth = videoStream.videoWidth;
    const videoHeight = videoStream.videoHeight;

    // Kích thước đầu vào cần thiết cho mô hình
    const targetSize = 224;

    // 1. Tính toán Tỷ lệ và Vị trí cắt (để lấy khung hình vuông ở giữa)
    let sx, sy, sWidth, sHeight; // source x, y, width, height
    
    if (videoWidth > videoHeight) {
        // Chiều rộng lớn hơn, cắt theo chiều rộng
        sWidth = sHeight = videoHeight;
        sx = (videoWidth - videoHeight) / 2;
        sy = 0;
    } else {
        // Chiều cao lớn hơn hoặc bằng, cắt theo chiều cao
        sWidth = sHeight = videoWidth;
        sx = 0;
        sy = (videoHeight - videoWidth) / 2;
    }

    // 2. Thiết lập Canvas và vẽ khung hình đã cắt
    cameraCanvas.width = targetSize;
    cameraCanvas.height = targetSize;
    const ctx = cameraCanvas.getContext('2d');
    
    // drawImage(source, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight)
    // Cắt (sx, sy, sWidth, sHeight) và thay đổi kích thước lên canvas (0, 0, targetSize, targetSize)
    ctx.drawImage(videoStream, sx, sy, sWidth, sHeight, 0, 0, targetSize, targetSize);

    // 3. Dừng camera
    stopCamera();

    // 4. Chuyển canvas thành Data URL và hiển thị
    const imageURL = cameraCanvas.toDataURL('image/jpeg');

    img.src = imageURL;
    img.style.display = 'block';
    
    // Reset thanh tiến trình
    progressBar.set(0); 

    // 5. Bắt đầu dự đoán
    document.querySelector('.init_status').style.display = 'block';
    boxResult.style.display = 'block';
    initialize().then( () => { 
        predict();
    });
}


// EVENT LISTENERS

activateCameraBtn.addEventListener('click', activateCamera);
captureImageBtn.addEventListener('click', captureImage);

fileUpload.addEventListener('change', function(e){
    
    // *Quan trọng*: Dừng camera nếu người dùng chuyển sang upload file
    stopCamera(); 

    let uploadedImage = e.target.value
    if (uploadedImage){
        document.getElementById("blankFile-1").innerHTML = uploadedImage.replace("C:\\fakepath\\","")
        document.getElementById("choose-text-1").innerText = "Đổi ảnh đã chọn" // Cập nhật text
        document.querySelector(".success-1").style.display = "inline-block"
        
        // Reset thanh tiến trình
        progressBar.set(0); 

        let extension = uploadedImage.split(".")[1]
        if (!(["doc","docx","pdf"].includes(extension))){
            document.querySelector(".success-1 i").style.border = "1px solid limegreen"
            document.querySelector(".success-1 i").style.color = "limegreen"
        }else{
            document.querySelector(".success-1 i").style.border = "1px solid rgb(25,110,180)"
            document.querySelector(".success-1 i").style.color = "rgb(25,110,180)"
        }
    }
    let file = this.files[0]
    if (file){
        document.querySelector('.init_status').style.display = 'block';
        boxResult.style.display = 'block'
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.addEventListener("load", function(){
            
            img.style.display = "block"
            img.setAttribute('src', this.result);
        });
    }

    else{
        img.setAttribute("src", "");
    }

    initialize().then( () => { 
        predict()
    })
})
