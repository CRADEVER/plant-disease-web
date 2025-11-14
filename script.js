// --- Model & App State ---
let model;
const CLASS_NAMES = [
  "Apple___Apple_scab",
  "Apple___Black_rot",
  "Apple___Cedar_apple_rust",
  "Apple___healthy",
  "Blueberry___healthy",
  // Add any other class names your model expects
];

// --- DOM Elements ---
const video = document.getElementById('camera');
const captureBtn = document.getElementById('capture');
const snapshotCanvas = document.getElementById('snapshot');
const resultDiv = document.getElementById('result');
const fileUploadInput = document.getElementById('file-upload');
const resetBtn = document.getElementById('reset-camera'); // Recommended: Add a reset button
const ctx = snapshotCanvas.getContext('2d');

// Disable buttons until model is loaded
captureBtn.disabled = true;
fileUploadInput.disabled = true;

// --- Core Functions ---

/**
 * Loads the TensorFlow.js model
 */
async function loadModel() {
  try {
    console.log('Loading model from: plant_model_js/model.json');
    resultDiv.innerText = 'Loading model... 🧠';
    model = await tf.loadLayersModel('plant_model_js/model.json');
    
    // Enable controls now that the model is loaded
    captureBtn.disabled = false;
    fileUploadInput.disabled = false;
    resultDiv.innerText = 'Model loaded. Camera ready. 📸';
  } catch (error) {
    console.error('Error loading model:', error);
    resultDiv.innerText = '❌ Error loading model.';
  }
}

/**
 * Starts the camera stream (using rear camera)
 * Also resets the view from a file upload.
 */
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'environment', // Use rear camera (from script 1)
        width: { ideal: 256 },
        height: { ideal: 256 }
      }
    });
    video.srcObject = stream;
    await video.play();

    // Show video, hide canvas
    video.style.display = 'block';
    snapshotCanvas.style.display = 'none';

    // Re-enable capture button if model is loaded
    if (model) {
      captureBtn.disabled = false;
      resultDiv.innerText = 'Camera ready. 📸';
    }
  } catch (error) {
    console.error('Error accessing camera:', error);
    resultDiv.innerText = '❌ Error accessing camera. Please grant permission.';
  }
}

/**
 * Runs the prediction on the image currently in the canvas
 */
async function predict() {
  try {
    const prediction = tf.tidy(() => {
      // Get image data from the canvas
      const img = tf.browser.fromPixels(snapshotCanvas)
        .resizeNearestNeighbor([256, 256]) // Model expects 256x256
        .toFloat()
        .sub(tf.scalar(127.5)) // Normalize pixel values
        .div(tf.scalar(127.5))
        .expandDims(); // Add batch dimension

      return model.predict(img);
    });

    const values = await prediction.data();
    const maxIndex = values.indexOf(Math.max(...values));
    const predictedClass = CLASS_NAMES[maxIndex];
    const confidence = (values[maxIndex] * 100).toFixed(2);

    // Display formatted result (from script 1)
    resultDiv.innerHTML = `
      🌳 <strong>Detected Disease:</strong> ${predictedClass}<br>
      📊 <strong>Confidence:</strong> ${confidence}%
    `;
  } catch (error) {
    console.error('Prediction error:', error);
    resultDiv.innerText = '❌ Error during prediction.';
  } finally {
    // Re-enable capture button (if we're in camera mode)
    if (video.style.display === 'block') {
      captureBtn.disabled = false;
    }
  }
}

// --- Event Listeners ---

/**
 * 1. Capture image from camera feed
 */
function captureImageFromCamera() {
  if (video.srcObject) {
    resultDiv.innerText = 'Analyzing... 🔍';
    captureBtn.disabled = true;

    // Draw the current video frame to the (hidden) canvas
    snapshotCanvas.width = video.videoWidth;
    snapshotCanvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);

    // Run prediction on the canvas image
    if (model) {
      predict();
    }
  } else {
    resultDiv.innerText = '⚠️ Camera not started.';
  }
}
captureBtn.addEventListener('click', captureImageFromCamera);

/**
 * 2. Handle file upload
 */
fileUploadInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();

    reader.onload = function(e) {
      const img = new Image();
      img.onload = function() {
        // --- This part is from Script 2 ---
        // Hide the video feed
        video.style.display = 'none';
        // Show the canvas and draw the uploaded image on it
        snapshotCanvas.style.display = 'block';
        snapshotCanvas.width = img.width;
        snapshotCanvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        // --- This part is from Script 1 ---
        // Now that the image is on the canvas, predict
        resultDiv.innerText = 'Analyzing uploaded image... 🔍';
        captureBtn.disabled = true; // Disable camera button
        if (model) {
          predict();
        }
      };
      img.src = e.target.result;
    };

    reader.readAsDataURL(file);
  }
});

/**
 * 3. (Recommended) Reset button to re-start camera
 */
if (resetBtn) {
  resetBtn.addEventListener('click', startCamera);
}


// --- Initial Startup ---
(async () => {
  // Start camera and load model at the same time
  await Promise.all([startCamera(), loadModel()]);
})();