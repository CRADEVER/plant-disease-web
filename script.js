let model;
const CLASS_NAMES = [
  "Apple___Apple_scab",
  "Apple___Black_rot",
  "Apple___Cedar_apple_rust",
  "Apple___healthy",
  "Blueberry___healthy",
  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
  "Corn_(maize)___Common_rust_",
  "Corn_(maize)___Northern_Leaf_Blight",
  "Corn_(maize)___healthy",
  "Grape___Black_rot",
  "Grape___Esca_(Black_Measles)",
  "Grape___healthy",
];

const video = document.getElementById('camera');
const captureBtn = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const ctx = snapshotCanvas.getContext('2d');

captureBtn.disabled = true;

async function loadModel() {
  try {
  console.log('Loading model from: plant_model_js/model.json');
  model = await tf.loadLayersModel('plant_model_js/model.json');
  captureBtn.disabled = false;
  resultDiv.innerText = 'Mô hình sẵn sàng! 👉 Hãy chụp ảnh để phân tích';
  } 
}

async function startCamera() {
try {
    const stream = await navigator.mediaDevices.getUserMedia({ 
    video: { facingMode: 'environment' }
    video.srcObject = stream;
    await video.play();
 } 
}

captureBtn.addEventListener('click', () => {
   resultDiv.innerText = 'Đang phân tích... 🔍';
   captureBtn.disabled = true;
   snapshotCanvas.width = video.videoWidth;
   snapshotCanvas.height = video.videoHeight;
ctx.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);
if (model) {
predict();
  } else {
    captureBtn.disabled = false;
   }
});

async function predict() { 
     tf.tidy(() => {
        const img = tf.browser.fromPixels(snapshotCanvas);
        const preprocessedImg = img
         .resizeNearestNeighbor([224, 224])
         .toFloat()
         .div(tf.scalar(255)) 
         .expandDims(); 
     const predictions = model.predict(preprocessedImg);
     const values = predictions.dataSync();
     const maxIndex = values.indexOf(Math.max(...values));
     const predictedClass = CLASS_NAMES[maxIndex];
     const confidence = (values[maxIndex] * 100).toFixed(2);
resultDiv.innerHTML = `
 🌳 **Bệnh được nhận diện:** **${predictedClass}**
<br>
  📊 **Độ tin cậy:** **${confidence}%**
 `;
 captureBtn.disabled = false;
  }); 
}
startCamera();
loadModel();