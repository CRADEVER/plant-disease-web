const video = document.getElementById('camera');
const captureButton = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const fileUploadInput = document.getElementById('file-upload');
const cameraSection = document.getElementById('camera-section'); 

async function startCamera() {
    try {
     
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        resultDiv.textContent = '✅ Camera đã sẵn sàng.';
    } catch (err) {
        console.error("Lỗi khi truy cập camera:", err);
        resultDiv.textContent = '❌ Không thể truy cập camera. Vui lòng kiểm tra quyền.';
        
        video.style.display = 'none'; 
    }
}

captureButton.addEventListener('click', () => {
   
    if (video.srcObject) {

        snapshotCanvas.width = video.videoWidth;
        snapshotCanvas.height = video.videoHeight;
     
        const context = snapshotCanvas.getContext('2d');
        context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        
        
        const imageDataURL = snapshotCanvas.toDataURL('image/jpeg');
        
        video.style.display = 'none';
        
     
        let previewImg = document.getElementById('uploaded-image-preview');
        if (!previewImg) {
            previewImg = document.createElement('img');
            previewImg.id = 'uploaded-image-preview';
            cameraSection.appendChild(previewImg);
        }
        previewImg.src = imageDataURL;
        previewImg.style.display = 'block';

        resultDiv.textContent = '📸 Ảnh đã chụp. Sẵn sàng phân tích.';
        
        
    } else {
        resultDiv.textContent = '⚠️ Camera chưa sẵn sàng hoặc đã bị tắt.';
    }
});

fileUploadInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
    
            video.style.display = 'none';
            
            let previewImg = document.getElementById('uploaded-image-preview');
if (!previewImg) {
                previewImg = document.createElement('img');
                previewImg.id = 'uploaded-image-preview';
                cameraSection.appendChild(previewImg);
            }
            
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            
            
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


window.onload = startCamera;