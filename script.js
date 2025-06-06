// Variables globales
let camera = null;
let pose = null;
let isRunning = false;
let showLandmarks = true;
let fpsCounter = 0;
let lastTime = performance.now();

// Variables para clasificación de movimientos
let isModelAvailable = false;
let currentPrediction = null;
let predictionHistory = [];
let lastPredictionTime = 0;
let lastValidPrediction = null; // Mantener la última predicción válida

// Elementos del DOM
const videoElement = document.getElementById("input_video");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const statusElement = document.getElementById("status");
const fpsElement = document.getElementById("fps");
const showLandmarksCheckbox = document.getElementById("show-landmarks");
const predictionElement = document.getElementById("prediction");
const confidenceElement = document.getElementById("confidence");

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

    // Verificar disponibilidad de la API del modelo
    await checkModelAPI();

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

    // Inicializar predicción por defecto
    if (!isModelAvailable) {
      lastValidPrediction = {
        prediction: "Esperando modelo...",
        confidence: 0.0,
      };
    }

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

  // Clasificar movimiento si el modelo está disponible
  if (isModelAvailable && results.poseLandmarks) {
    classifyMovement(results.poseLandmarks);
  }

  // Siempre mostrar la última predicción disponible
  if (lastValidPrediction) {
    displayPredictionOnCanvas(lastValidPrediction);
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
  if (statusElement) {
    statusElement.textContent = message;
    statusElement.parentElement.className = `status card ${type}`;
  }

  // Actualizar también el indicador de cámara en el video (solo si existe)
  const cameraStatus = document.getElementById("camera-status");
  if (cameraStatus) {
    const cameraIcon = cameraStatus.querySelector("i");
    const cameraText = cameraStatus.querySelector("span");

    if (cameraIcon && cameraText) {
      if (type === "connected") {
        cameraIcon.setAttribute("data-lucide", "camera");
        cameraText.textContent = "Cámara activa";
      } else if (type === "error") {
        cameraIcon.setAttribute("data-lucide", "camera-off");
        cameraText.textContent = "Error de cámara";
      } else if (type === "loading") {
        cameraIcon.setAttribute("data-lucide", "loader");
        cameraText.textContent = "Conectando...";
      } else {
        cameraIcon.setAttribute("data-lucide", "camera-off");
        cameraText.textContent = "Cámara desconectada";
      }

      // Actualizar iconos
      if (typeof lucide !== "undefined") {
        lucide.createIcons();
      }
    }
  }
}

// Función para actualizar FPS
function updateFPS() {
  fpsCounter++;
  const currentTime = performance.now();

  if (currentTime - lastTime >= 1000) {
    fpsElement.textContent = `${fpsCounter} FPS`;
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

// ========== FUNCIONES DE CLASIFICACIÓN DE MOVIMIENTOS ==========

// Verificar disponibilidad de la API del modelo
async function checkModelAPI() {
  const modelStatusElement = document.getElementById("model-status");

  try {
    const response = await fetch("http://localhost:5000/health");
    if (response.ok) {
      const data = await response.json();
      isModelAvailable = data.model_loaded;

      if (isModelAvailable) {
        modelStatusElement.textContent = `${
          data.model_info?.name || "Cargado"
        } ✅`;
        modelStatusElement.parentElement.className =
          "model-status card available";
        console.log("✅ API del modelo disponible:", data);

        // Establecer predicción inicial
        lastValidPrediction = {
          prediction: "Detectando movimiento...",
          confidence: 0.0,
        };
      } else {
        modelStatusElement.textContent = "Error de carga ❌";
        modelStatusElement.parentElement.className =
          "model-status card unavailable";
      }
      return isModelAvailable;
    }
  } catch (error) {
    console.log("⚠️ API del modelo no disponible:", error.message);
  }

  isModelAvailable = false;
  modelStatusElement.textContent = "No disponible ❌";
  modelStatusElement.parentElement.className = "model-status card unavailable";

  // Establecer predicción por defecto
  lastValidPrediction = {
    prediction: "Modelo no disponible",
    confidence: 0.0,
  };

  return false;
}

// Clasificar movimiento usando la API
async function classifyMovement(landmarks) {
  // Limitar frecuencia de predicciones (máximo cada 200ms para mejor fluidez)
  const now = Date.now();
  if (now - lastPredictionTime < 200) {
    return;
  }
  lastPredictionTime = now;

  try {
    // Índices de landmarks relevantes para el modelo
    const relevantLandmarkIndices = [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    ];

    // Preparar datos de landmarks - solo los relevantes con landmark_index
    const landmarksData = relevantLandmarkIndices
      .map((index) => {
        const landmark = landmarks[index];
        if (landmark) {
          return {
            landmark_index: index,
            x: landmark.x,
            y: landmark.y,
            z: landmark.z,
            visibility: landmark.visibility,
          };
        }
        return null;
      })
      .filter((landmark) => landmark !== null); // Filtrar landmarks nulos

    // Verificar que tenemos al menos algunos landmarks
    if (landmarksData.length === 0) {
      console.warn("No se encontraron landmarks relevantes para clasificación");
      return;
    }

    console.log(
      `Enviando ${landmarksData.length} landmarks relevantes a la API`
    );

    // Enviar a la API
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        landmarks: landmarksData,
      }),
    });

    if (response.ok) {
      const result = await response.json();
      updatePrediction(result);
    } else {
      console.error("Error en predicción:", response.statusText);
    }
  } catch (error) {
    console.error("Error clasificando movimiento:", error);
  }
}

// Actualizar la predicción mostrada
function updatePrediction(result) {
  if (result.error) {
    console.error("Error del modelo:", result.error);
    return;
  }

  currentPrediction = result;
  lastValidPrediction = result; // Almacenar la última predicción válida

  // Agregar a historial
  predictionHistory.push({
    prediction: result.prediction,
    confidence: result.confidence,
    timestamp: Date.now(),
  });

  // Mantener solo las últimas 10 predicciones
  if (predictionHistory.length > 10) {
    predictionHistory.shift();
  }

  // La predicción se mostrará en onResults(), no aquí para evitar duplicados
}

// Mostrar predicción en el canvas
function displayPredictionOnCanvas(result) {
  if (!canvasCtx || !result) return;

  // Configurar estilo del texto
  canvasCtx.font = "bold 24px Arial";
  canvasCtx.fillStyle = "#FFFFFF";
  canvasCtx.strokeStyle = "#000000";
  canvasCtx.lineWidth = 3;

  // Texto de la predicción
  const predictionText = `Movimiento: ${result.prediction}`;
  const confidenceText = `Confianza: ${(result.confidence * 100).toFixed(1)}%`;

  // Posición del texto
  const x = 20;
  const y = 40;

  // Dibujar fondo semi-transparente más visible
  canvasCtx.fillStyle = "rgba(0, 0, 0, 0.8)";
  canvasCtx.fillRect(10, 10, 380, 90);

  // Dibujar borde del rectángulo para mayor visibilidad
  canvasCtx.strokeStyle = "#FFFFFF";
  canvasCtx.lineWidth = 2;
  canvasCtx.strokeRect(10, 10, 380, 90);

  // Dibujar texto con borde
  canvasCtx.fillStyle = "#FFFFFF";
  canvasCtx.strokeStyle = "#000000";
  canvasCtx.lineWidth = 3;
  canvasCtx.strokeText(predictionText, x, y);
  canvasCtx.fillText(predictionText, x, y);

  canvasCtx.strokeText(confidenceText, x, y + 30);
  canvasCtx.fillText(confidenceText, x, y + 30);

  // Cambiar color según confianza para la barra
  const confidence = result.confidence;
  let barColor;
  if (confidence > 0.8) {
    barColor = "#00FF00"; // Verde para alta confianza
  } else if (confidence > 0.6) {
    barColor = "#FFFF00"; // Amarillo para confianza media
  } else {
    barColor = "#FF6600"; // Naranja para baja confianza
  }

  // Barra de confianza
  const barWidth = 300;
  const barHeight = 12;
  const barX = x;
  const barY = y + 45;

  // Fondo de la barra
  canvasCtx.fillStyle = "rgba(255, 255, 255, 0.3)";
  canvasCtx.fillRect(barX, barY, barWidth, barHeight);

  // Borde de la barra
  canvasCtx.strokeStyle = "#FFFFFF";
  canvasCtx.lineWidth = 1;
  canvasCtx.strokeRect(barX, barY, barWidth, barHeight);

  // Barra de progreso
  canvasCtx.fillStyle = barColor;
  canvasCtx.fillRect(barX, barY, barWidth * confidence, barHeight);

  // Mostrar timestamp para debug (opcional)
  const timeText = `Actualizado: ${new Date().toLocaleTimeString()}`;
  canvasCtx.font = "12px Arial";
  canvasCtx.fillStyle = "#CCCCCC";
  canvasCtx.fillText(timeText, x, y + 75);
}
