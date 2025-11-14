const video = document.getElementById('camera');
const captureButton = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const fileUploadInput = document.getElementById('file-upload');
const cameraSection = document.getElementById('camera-section'); // Thêm ID mới

// 1. Khởi động Camera
async function startCamera() {
    try {
        // Yêu cầu quyền truy cập camera, chỉ lấy video
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        resultDiv.textContent = '✅ Camera đã sẵn sàng.';
    } catch (err) {
        console.error("Lỗi khi truy cập camera:", err);
        resultDiv.textContent = '❌ Không thể truy cập camera. Vui lòng kiểm tra quyền.';
        // Ẩn video nếu không thể khởi động
        video.style.display = 'none'; 
    }
}

// 2. Xử lý Chụp ảnh từ Camera
captureButton.addEventListener('click', () => {
    // Đảm bảo video đang hiển thị và có luồng dữ liệu
    if (video.srcObject) {
        // Thiết lập kích thước canvas bằng kích thước video
        snapshotCanvas.width = video.videoWidth;
        snapshotCanvas.height = video.videoHeight;
        
        // Vẽ khung hình hiện tại của video lên canvas
        const context = snapshotCanvas.getContext('2d');
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        
        // Hiển thị ảnh vừa chụp (thay thế video)
        const imageDataURL = snapshotCanvas.toDataURL('image/jpeg');
        
        // Tạm dừng luồng video
        video.style.display = 'none';
        
        // Tạo một thẻ <img> để hiển thị ảnh chụp (giữ nguyên layout)
        let previewImg = document.getElementById('uploaded-image-preview');
        if (!previewImg) {
            previewImg = document.createElement('img');
            previewImg.id = 'uploaded-image-preview';
            cameraSection.appendChild(previewImg);
        }
        previewImg.src = imageDataURL;
        previewImg.style.display = 'block';

        resultDiv.textContent = '📸 Ảnh đã chụp. Sẵn sàng phân tích.';
        
        // Tắt camera (optional, để tiết kiệm pin)
        // video.srcObject.getTracks().forEach(track => track.stop());
    } else {
        resultDiv.textContent = '⚠️ Camera chưa sẵn sàng hoặc đã bị tắt.';
    }
});


// 3. Xử lý Tải ảnh lên (Thêm ảnh)
fileUploadInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            // Hiển thị ảnh tải lên (thay thế video)
            video.style.display = 'none';
            
            let previewImg = document.getElementById('uploaded-image-preview');
if (!previewImg) {
                previewImg = document.createElement('img');
                previewImg.id = 'uploaded-image-preview';
                cameraSection.appendChild(previewImg);
            }
            
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            
            // Vẽ ảnh tải lên lên canvas (chuẩn bị cho phân tích)
            const img = new Image();
            img.onload = function() {
                snapshotCanvas.width = img.width;
                snapshotCanvas.height = img.height;
                const context = snapshotCanvas.getContext('2d');
                context.drawImage(img, 0, 0);
            };
            img.src = e.target.result;

            resultDiv.textContent = `⬆️ Đã tải lên "${file.name}". Sẵn sàng phân tích.`;
        };
        
        reader.readAsDataURL(file);
    }
});

// Bắt đầu camera khi trang web được tải
window.onload = startCamera;