// Variables globales
let camera = null;
let pose = null;
let isRunning = false;
let showLandmarks = true;
let fpsCounter = 0;
let lastTime = performance.now();

// Elementos del DOM
const videoElement = document.getElementById("input_video");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const statusElement = document.getElementById("status");
const fpsElement = document.getElementById("fps");
const showLandmarksCheckbox = document.getElementById("show-landmarks");

// Configuración de MediaPipe Pose
const poseConfig = {
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
  },
};

// Inicialización de la aplicación
async function initializeApp() {
  try {
    updateStatus("Inicializando...", "loading");

    // Inicializar MediaPipe Pose
    pose = new Pose(poseConfig);
    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults(onResults);

    updateStatus("Listo para iniciar", "ready");
  } catch (error) {
    console.error("Error al inicializar:", error);
    updateStatus("Error de inicialización", "error");
  }
}

// Función para manejar los resultados de MediaPipe
function onResults(results) {
  if (!canvasElement || !canvasCtx) return;

  // Actualizar FPS
  updateFPS();

  // Configurar canvas
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // Dibujar la imagen del video (opcional, para debug)
  // canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  // Dibujar landmarks si están habilitados
  if (showLandmarks && results.poseLandmarks) {
    drawPoseLandmarks(results.poseLandmarks);
    drawPoseConnections(results.poseLandmarks);
  }

  canvasCtx.restore();
}

// Función para dibujar los landmarks de pose
function drawPoseLandmarks(landmarks) {
  if (!landmarks) return;

  // Configuración de estilo para los landmarks
  canvasCtx.fillStyle = "#FF0000";
  canvasCtx.strokeStyle = "#FF0000";
  canvasCtx.lineWidth = 2;

  // Dibujar cada landmark
  landmarks.forEach((landmark, index) => {
    if (landmark.visibility > 0.5) {
      const x = landmark.x * canvasElement.width;
      const y = landmark.y * canvasElement.height;

      // Dibujar punto
      canvasCtx.beginPath();
      canvasCtx.arc(x, y, 4, 0, 2 * Math.PI);
      canvasCtx.fill();
    }
  });
}

// Función para dibujar las conexiones entre landmarks
function drawPoseConnections(landmarks) {
  if (!landmarks) return;

  // Definir conexiones principales del cuerpo
  const connections = [
    // Cara
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 7],
    [0, 4],
    [4, 5],
    [5, 6],
    [6, 8],

    // Brazos
    [9, 10],
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
    [15, 17],
    [15, 19],
    [15, 21],
    [16, 18],
    [16, 20],
    [16, 22],

    // Torso
    [11, 23],
    [12, 24],
    [23, 24],

    // Piernas
    [23, 25],
    [24, 26],
    [25, 27],
    [26, 28],
    [27, 29],
    [28, 30],
    [29, 31],
    [30, 32],
    [27, 31],
    [28, 32],
  ];

  canvasCtx.strokeStyle = "#00FF00";
  canvasCtx.lineWidth = 2;

  connections.forEach(([startIdx, endIdx]) => {
    const startPoint = landmarks[startIdx];
    const endPoint = landmarks[endIdx];

    if (
      startPoint &&
      endPoint &&
      startPoint.visibility > 0.5 &&
      endPoint.visibility > 0.5
    ) {
      const startX = startPoint.x * canvasElement.width;
      const startY = startPoint.y * canvasElement.height;
      const endX = endPoint.x * canvasElement.width;
      const endY = endPoint.y * canvasElement.height;

      canvasCtx.beginPath();
      canvasCtx.moveTo(startX, startY);
      canvasCtx.lineTo(endX, endY);
      canvasCtx.stroke();
    }
  });
}

// Función para iniciar la cámara
async function startCamera() {
  try {
    updateStatus("Iniciando cámara...", "loading");

    // Configurar cámara
    camera = new Camera(videoElement, {
      onFrame: async () => {
        if (pose && isRunning) {
          await pose.send({ image: videoElement });
        }
      },
      width: 640,
      height: 480,
    });

    // Iniciar cámara
    await camera.start();

    // Configurar canvas
    canvasElement.width = videoElement.videoWidth || 640;
    canvasElement.height = videoElement.videoHeight || 480;

    isRunning = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    videoElement.classList.add("recording");

    updateStatus("Cámara activa", "connected");
  } catch (error) {
    console.error("Error al iniciar cámara:", error);
    updateStatus("Error al acceder a la cámara", "error");

    // Mostrar mensaje más específico al usuario
    if (error.name === "NotAllowedError") {
      alert(
        "Acceso a la cámara denegado. Por favor, permite el acceso a la cámara y recarga la página."
      );
    } else if (error.name === "NotFoundError") {
      alert(
        "No se encontró ninguna cámara. Asegúrate de tener una cámara conectada."
      );
    } else {
      alert("Error al acceder a la cámara: " + error.message);
    }
  }
}

// Función para detener la cámara
function stopCamera() {
  try {
    if (camera) {
      camera.stop();
      camera = null;
    }

    isRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    videoElement.classList.remove("recording");

    // Limpiar canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    updateStatus("Cámara detenida", "ready");
  } catch (error) {
    console.error("Error al detener cámara:", error);
    updateStatus("Error al detener cámara", "error");
  }
}

// Función para actualizar el estado
function updateStatus(message, type = "default") {
  statusElement.textContent = `Estado: ${message}`;
  statusElement.parentElement.className = `status ${type}`;
}

// Función para actualizar FPS
function updateFPS() {
  fpsCounter++;
  const currentTime = performance.now();

  if (currentTime - lastTime >= 1000) {
    fpsElement.textContent = `FPS: ${fpsCounter}`;
    fpsCounter = 0;
    lastTime = currentTime;
  }
}

// Función para toggle de landmarks
function toggleLandmarks() {
  showLandmarks = showLandmarksCheckbox.checked;
}

// Event listeners
document.addEventListener("DOMContentLoaded", () => {
  initializeApp();

  startBtn.addEventListener("click", startCamera);
  stopBtn.addEventListener("click", stopCamera);
  showLandmarksCheckbox.addEventListener("change", toggleLandmarks);

  // Manejar cambio de tamaño de ventana
  window.addEventListener("resize", () => {
    if (isRunning && videoElement.videoWidth && videoElement.videoHeight) {
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
    }
  });

  // Manejar visibilidad de la página
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && isRunning) {
      // Pausar cuando la página no es visible para ahorrar recursos
      isRunning = false;
    } else if (!document.hidden && camera) {
      // Reanudar cuando la página vuelve a ser visible
      isRunning = true;
    }
  });
});

// Manejar errores globales
window.addEventListener("error", (event) => {
  console.error("Error global:", event.error);
  updateStatus("Error inesperado", "error");
});

// Función de utilidad para verificar soporte del navegador
function checkBrowserSupport() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert(
      "Tu navegador no soporta acceso a la cámara. Por favor, usa un navegador moderno."
    );
    return false;
  }

  if (!window.MediaPipe) {
    console.warn("MediaPipe no está disponible");
  }

  return true;
}

// Verificar soporte al cargar
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", checkBrowserSupport);
} else {
  checkBrowserSupport();
}
