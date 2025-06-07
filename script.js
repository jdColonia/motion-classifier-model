// Variables globales
let camera = null
let pose = null
let isRunning = false
let showLandmarks = true
let fpsCounter = 0
let lastTime = performance.now()

// Variables para clasificación de movimientos
let isModelAvailable = false
let currentPrediction = null
let predictionHistory = []
let lastPredictionTime = 0
let lastValidPrediction = null
let predictionQueue = [] // Cola para suavizar predicciones

// Elementos del DOM
const videoElement = document.getElementById('input_video')
const canvasElement = document.getElementById('output_canvas')
const canvasCtx = canvasElement.getContext('2d')
const startBtn = document.getElementById('start-btn')
const stopBtn = document.getElementById('stop-btn')
const statusElement = document.getElementById('status')
const fpsElement = document.getElementById('fps')
const showLandmarksCheckbox = document.getElementById('show-landmarks')

// Configuración de MediaPipe Pose
const poseConfig = {
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
  },
}

// Inicialización de la aplicación
async function initializeApp() {
  try {
    updateStatus('Inicializando...', 'loading')

    // Verificar disponibilidad de la API del modelo
    await checkModelAPI()

    // Inicializar MediaPipe Pose
    pose = new Pose(poseConfig)
    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      smoothSegmentation: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    })

    pose.onResults(onResults)

    // Inicializar predicción por defecto
    if (!isModelAvailable) {
      lastValidPrediction = {
        prediction: 'Modelo no disponible',
        confidence: 0.0,
      }
    } else {
      lastValidPrediction = {
        prediction: 'Esperando detección...',
        confidence: 0.0,
      }
    }

    updateStatus('Listo para iniciar', 'ready')
  } catch (error) {
    console.error('Error al inicializar:', error)
    updateStatus('Error de inicialización', 'error')
  }
}

// Función para manejar los resultados de MediaPipe
function onResults(results) {
  if (!canvasElement || !canvasCtx) return

  // Actualizar FPS
  updateFPS()

  // Configurar canvas
  canvasCtx.save()
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height)

  // Dibujar landmarks si están habilitados
  if (showLandmarks && results.poseLandmarks) {
    drawPoseLandmarks(results.poseLandmarks)
    drawPoseConnections(results.poseLandmarks)
  }

  // Clasificar movimiento si el modelo está disponible y hay landmarks
  if (isModelAvailable && results.poseLandmarks && results.poseLandmarks.length > 0) {
    classifyMovement(results.poseLandmarks)
  }

  // Mostrar predicción (siempre mostrar la última válida)
  if (lastValidPrediction) {
    displayPredictionOnCanvas(lastValidPrediction)
  }

  canvasCtx.restore()
}

// Función para dibujar los landmarks de pose
function drawPoseLandmarks(landmarks) {
  if (!landmarks || landmarks.length === 0) return

  // Configuración de estilo para los landmarks
  canvasCtx.fillStyle = '#FF0000'
  canvasCtx.strokeStyle = '#FF0000'
  canvasCtx.lineWidth = 2

  // Dibujar cada landmark
  landmarks.forEach((landmark, index) => {
    if (landmark.visibility > 0.5) {
      const x = landmark.x * canvasElement.width
      const y = landmark.y * canvasElement.height

      // Dibujar punto
      canvasCtx.beginPath()
      canvasCtx.arc(x, y, 4, 0, 2 * Math.PI)
      canvasCtx.fill()
    }
  })
}

// Función para dibujar las conexiones entre landmarks
function drawPoseConnections(landmarks) {
  if (!landmarks || landmarks.length === 0) return

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
  ]

  canvasCtx.strokeStyle = '#00FF00'
  canvasCtx.lineWidth = 2

  connections.forEach(([startIdx, endIdx]) => {
    const startPoint = landmarks[startIdx]
    const endPoint = landmarks[endIdx]

    if (startPoint && endPoint && startPoint.visibility > 0.5 && endPoint.visibility > 0.5) {
      const startX = startPoint.x * canvasElement.width
      const startY = startPoint.y * canvasElement.height
      const endX = endPoint.x * canvasElement.width
      const endY = endPoint.y * canvasElement.height

      canvasCtx.beginPath()
      canvasCtx.moveTo(startX, startY)
      canvasCtx.lineTo(endX, endY)
      canvasCtx.stroke()
    }
  })
}

// Función para iniciar la cámara
async function startCamera() {
  try {
    updateStatus('Iniciando cámara...', 'loading')

    // Configurar cámara
    camera = new Camera(videoElement, {
      onFrame: async () => {
        if (pose && isRunning) {
          await pose.send({ image: videoElement })
        }
      },
      width: 640,
      height: 480,
    })

    // Iniciar cámara
    await camera.start()

    // Configurar canvas
    canvasElement.width = videoElement.videoWidth || 640
    canvasElement.height = videoElement.videoHeight || 480

    isRunning = true
    startBtn.disabled = true
    stopBtn.disabled = false
    videoElement.classList.add('recording')

    updateStatus('Cámara activa', 'connected')
  } catch (error) {
    console.error('Error al iniciar cámara:', error)
    updateStatus('Error al acceder a la cámara', 'error')

    // Mostrar mensaje más específico al usuario
    if (error.name === 'NotAllowedError') {
      alert('Acceso a la cámara denegado. Por favor, permite el acceso a la cámara y recarga la página.')
    } else if (error.name === 'NotFoundError') {
      alert('No se encontró ninguna cámara. Asegúrate de tener una cámara conectada.')
    } else {
      alert('Error al acceder a la cámara: ' + error.message)
    }
  }
}

// Función para detener la cámara
function stopCamera() {
  try {
    if (camera) {
      camera.stop()
      camera = null
    }

    isRunning = false
    startBtn.disabled = false
    stopBtn.disabled = true
    videoElement.classList.remove('recording')

    // Limpiar canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height)

    updateStatus('Cámara detenida', 'ready')
  } catch (error) {
    console.error('Error al detener cámara:', error)
    updateStatus('Error al detener cámara', 'error')
  }
}

// Función para actualizar el estado
function updateStatus(message, type = 'default') {
  if (statusElement) {
    statusElement.textContent = message
    statusElement.parentElement.className = `status card ${type}`
  }

  // Actualizar también el indicador de cámara en el video
  const cameraStatus = document.getElementById('camera-status')
  if (cameraStatus) {
    const cameraIcon = cameraStatus.querySelector('i')
    const cameraText = cameraStatus.querySelector('span')

    if (cameraIcon && cameraText) {
      if (type === 'connected') {
        cameraIcon.setAttribute('data-lucide', 'camera')
        cameraText.textContent = 'Cámara activa'
      } else if (type === 'error') {
        cameraIcon.setAttribute('data-lucide', 'camera-off')
        cameraText.textContent = 'Error de cámara'
      } else if (type === 'loading') {
        cameraIcon.setAttribute('data-lucide', 'loader')
        cameraText.textContent = 'Conectando...'
      } else {
        cameraIcon.setAttribute('data-lucide', 'camera-off')
        cameraText.textContent = 'Cámara desconectada'
      }

      // Actualizar iconos
      if (typeof lucide !== 'undefined') {
        lucide.createIcons()
      }
    }
  }
}

// Función para actualizar FPS
function updateFPS() {
  fpsCounter++
  const currentTime = performance.now()

  if (currentTime - lastTime >= 1000) {
    fpsElement.textContent = `${fpsCounter} FPS`
    fpsCounter = 0
    lastTime = currentTime
  }
}

// Función para toggle de landmarks
function toggleLandmarks() {
  showLandmarks = showLandmarksCheckbox.checked
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
  initializeApp()

  startBtn.addEventListener('click', startCamera)
  stopBtn.addEventListener('click', stopCamera)
  showLandmarksCheckbox.addEventListener('change', toggleLandmarks)

  // Manejar cambio de tamaño de ventana
  window.addEventListener('resize', () => {
    if (isRunning && videoElement.videoWidth && videoElement.videoHeight) {
      canvasElement.width = videoElement.videoWidth
      canvasElement.height = videoElement.videoHeight
    }
  })

  // Manejar visibilidad de la página
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && isRunning) {
      // Pausar cuando la página no es visible para ahorrar recursos
      isRunning = false
    } else if (!document.hidden && camera) {
      // Reanudar cuando la página vuelve a ser visible
      isRunning = true
    }
  })
})

// Manejar errores globales
window.addEventListener('error', (event) => {
  console.error('Error global:', event.error)
  updateStatus('Error inesperado', 'error')
})

// Función de utilidad para verificar soporte del navegador
function checkBrowserSupport() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Tu navegador no soporta acceso a la cámara. Por favor, usa un navegador moderno.')
    return false
  }

  if (!window.MediaPipe) {
    console.warn('MediaPipe no está disponible')
  }

  return true
}

// Verificar soporte al cargar
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', checkBrowserSupport)
} else {
  checkBrowserSupport()
}

// ========== FUNCIONES DE CLASIFICACIÓN DE MOVIMIENTOS ==========

// Verificar disponibilidad de la API del modelo
async function checkModelAPI() {
  const modelStatusElement = document.getElementById('model-status')

  try {
    // Intentar conectar con timeout más corto
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 3000) // 3 segundos timeout

    const response = await fetch('http://localhost:5000/health', {
      signal: controller.signal,
    })

    clearTimeout(timeoutId)

    if (response.ok) {
      const data = await response.json()
      isModelAvailable = data.model_loaded

      if (isModelAvailable) {
        modelStatusElement.textContent = `${data.model_info?.name || 'Cargado'} ✅`
        modelStatusElement.parentElement.className = 'model-status card available'
        console.log('✅ API del modelo disponible:', data)

        // Establecer predicción inicial
        lastValidPrediction = {
          prediction: 'Detectando movimiento...',
          confidence: 0.0,
        }
      } else {
        modelStatusElement.textContent = 'Error de carga ❌'
        modelStatusElement.parentElement.className = 'model-status card unavailable'
      }
      return isModelAvailable
    }
  } catch (error) {
    console.log('⚠️ API del modelo no disponible:', error.message)
  }

  isModelAvailable = false
  modelStatusElement.textContent = 'No disponible ❌'
  modelStatusElement.parentElement.className = 'model-status card unavailable'

  // Establecer predicción por defecto
  lastValidPrediction = {
    prediction: 'Modelo no disponible',
    confidence: 0.0,
  }

  return false
}

// Clasificar movimiento usando la API MEJORADA
async function classifyMovement(landmarks) {
  // Limitar frecuencia de predicciones (máximo cada 300ms para estabilidad)
  const now = Date.now()
  if (now - lastPredictionTime < 300) {
    return
  }
  lastPredictionTime = now

  try {
    // Validar que tenemos landmarks válidos
    if (!landmarks || landmarks.length === 0) {
      console.warn('No se recibieron landmarks para clasificación')
      return
    }

    // Mapear landmarks de MediaPipe a los índices que espera el modelo
    const relevantLandmarkIndices = [
      0, // nose
      11,
      12, // shoulders
      13,
      14, // elbows
      15,
      16, // wrists
      23,
      24, // hips
      25,
      26, // knees
      27,
      28, // ankles
      31,
      32, // feet
    ]

    // Preparar datos con el formato correcto
    const landmarksData = []

    relevantLandmarkIndices.forEach((index) => {
      if (index < landmarks.length) {
        const landmark = landmarks[index]
        if (landmark && landmark.visibility > 0.3) {
          // Umbral más bajo para incluir más landmarks
          landmarksData.push({
            landmark_index: index,
            x: Number(landmark.x),
            y: Number(landmark.y),
            z: Number(landmark.z || 0), // z puede no estar disponible
            visibility: Number(landmark.visibility),
          })
        }
      }
    })

    // Verificar que tenemos landmarks suficientes
    if (landmarksData.length < 8) {
      // Mínimo 8 landmarks para una predicción decente
      console.warn(`Landmarks insuficientes para clasificación: ${landmarksData.length}`)

      // Actualizar predicción con estado de landmarks insuficientes
      lastValidPrediction = {
        prediction: 'Landmarks insuficientes',
        confidence: 0.0,
      }
      return
    }

    console.log(`Enviando ${landmarksData.length} landmarks válidos a la API`)

    // Enviar a la API con timeout
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 segundos timeout

    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        landmarks: landmarksData,
      }),
      signal: controller.signal,
    })

    clearTimeout(timeoutId)

    if (response.ok) {
      const result = await response.json()

      if (result.error) {
        console.error('Error del modelo:', result.error)
        lastValidPrediction = {
          prediction: 'Error en predicción',
          confidence: 0.0,
        }
        return
      }

      // Suavizar predicciones usando cola
      updatePredictionWithSmoothing(result)
    } else {
      console.error('Error HTTP en predicción:', response.status, response.statusText)
      const errorText = await response.text()
      console.error('Detalle del error:', errorText)
    }
  } catch (error) {
    if (error.name === 'AbortError') {
      console.warn('Timeout en predicción - la API está sobrecargada')
    } else {
      console.error('Error clasificando movimiento:', error)
    }

    // Mantener la última predicción válida en caso de error
    if (!lastValidPrediction) {
      lastValidPrediction = {
        prediction: 'Error de conexión',
        confidence: 0.0,
      }
    }
  }
}

// Nueva función para suavizar predicciones
function updatePredictionWithSmoothing(result) {
  // Agregar a la cola de predicciones
  predictionQueue.push({
    prediction: result.prediction,
    confidence: result.confidence,
    timestamp: Date.now(),
  })

  // Mantener solo las últimas 5 predicciones
  if (predictionQueue.length > 5) {
    predictionQueue.shift()
  }

  // Calcular predicción suavizada
  const smoothedPrediction = calculateSmoothedPrediction()

  // Actualizar predicción actual
  currentPrediction = result
  lastValidPrediction = smoothedPrediction

  // Agregar a historial
  predictionHistory.push({
    prediction: smoothedPrediction.prediction,
    confidence: smoothedPrediction.confidence,
    timestamp: Date.now(),
  })

  // Mantener solo las últimas 20 predicciones en el historial
  if (predictionHistory.length > 20) {
    predictionHistory.shift()
  }

  console.log(
    `Predicción suavizada: ${smoothedPrediction.prediction} (confianza: ${smoothedPrediction.confidence.toFixed(3)})`
  )
}

// Calcular predicción suavizada
function calculateSmoothedPrediction() {
  if (predictionQueue.length === 0) {
    return {
      prediction: 'Esperando...',
      confidence: 0.0,
    }
  }

  // Contar ocurrencias de cada predicción
  const predictionCounts = {}
  let totalConfidence = 0

  predictionQueue.forEach((pred) => {
    if (!predictionCounts[pred.prediction]) {
      predictionCounts[pred.prediction] = {
        count: 0,
        totalConfidence: 0,
      }
    }
    predictionCounts[pred.prediction].count++
    predictionCounts[pred.prediction].totalConfidence += pred.confidence
    totalConfidence += pred.confidence
  })

  // Encontrar la predicción más frecuente
  let mostFrequent = null
  let maxCount = 0

  for (const [prediction, data] of Object.entries(predictionCounts)) {
    if (data.count > maxCount) {
      maxCount = data.count
      mostFrequent = prediction
    }
  }

  // Calcular confianza promedio para la predicción más frecuente
  const avgConfidence = predictionCounts[mostFrequent].totalConfidence / predictionCounts[mostFrequent].count

  return {
    prediction: mostFrequent,
    confidence: avgConfidence,
  }
}

// Mostrar predicción en el canvas (MEJORADA)
function displayPredictionOnCanvas(result) {
  if (!canvasCtx || !result) return

  // Configurar estilo del texto
  canvasCtx.font = 'bold 28px Arial'
  canvasCtx.textAlign = 'left'

  // Texto de la predicción
  const predictionText = `${result.prediction}`
  const confidenceText = `Confianza: ${(result.confidence * 100).toFixed(1)}%`

  // Posición del texto
  const x = 20
  const y = 50

  // Determinar color según confianza
  let backgroundColor, textColor, borderColor
  const confidence = result.confidence

  if (confidence > 0.8) {
    backgroundColor = 'rgba(0, 150, 0, 0.9)' // Verde para alta confianza
    textColor = '#FFFFFF'
    borderColor = '#00FF00'
  } else if (confidence > 0.6) {
    backgroundColor = 'rgba(255, 165, 0, 0.9)' // Naranja para confianza media
    textColor = '#FFFFFF'
    borderColor = '#FFAA00'
  } else if (confidence > 0.3) {
    backgroundColor = 'rgba(255, 255, 0, 0.9)' // Amarillo para confianza baja
    textColor = '#000000'
    borderColor = '#FFFF00'
  } else {
    backgroundColor = 'rgba(150, 0, 0, 0.9)' // Rojo para muy baja confianza
    textColor = '#FFFFFF'
    borderColor = '#FF0000'
  }

  // Dibujar fondo con color dinámico
  canvasCtx.fillStyle = backgroundColor
  canvasCtx.fillRect(10, 10, 420, 100)

  // Dibujar borde del rectángulo
  canvasCtx.strokeStyle = borderColor
  canvasCtx.lineWidth = 3
  canvasCtx.strokeRect(10, 10, 420, 100)

  // Dibujar texto principal
  canvasCtx.fillStyle = textColor
  canvasCtx.strokeStyle = 'rgba(0, 0, 0, 0.7)'
  canvasCtx.lineWidth = 2

  // Texto con sombra para mejor legibilidad
  canvasCtx.strokeText(predictionText, x, y)
  canvasCtx.fillText(predictionText, x, y)

  // Texto de confianza más pequeño
  canvasCtx.font = 'bold 20px Arial'
  canvasCtx.strokeText(confidenceText, x, y + 35)
  canvasCtx.fillText(confidenceText, x, y + 35)

  // Barra de confianza mejorada
  const barWidth = 380
  const barHeight = 15
  const barX = x
  const barY = y + 50

  // Fondo de la barra
  canvasCtx.fillStyle = 'rgba(255, 255, 255, 0.4)'
  canvasCtx.fillRect(barX, barY, barWidth, barHeight)

  // Borde de la barra
  canvasCtx.strokeStyle = textColor
  canvasCtx.lineWidth = 2
  canvasCtx.strokeRect(barX, barY, barWidth, barHeight)

  // Barra de progreso con gradiente
  const gradient = canvasCtx.createLinearGradient(barX, barY, barX + barWidth * confidence, barY)
  gradient.addColorStop(0, borderColor)
  gradient.addColorStop(1, textColor)

  canvasCtx.fillStyle = gradient
  canvasCtx.fillRect(barX, barY, barWidth * confidence, barHeight)

  // Mostrar timestamp (más pequeño)
  const timeText = `${new Date().toLocaleTimeString()}`
  canvasCtx.font = '12px Arial'
  canvasCtx.fillStyle = textColor
  canvasCtx.fillText(timeText, x, y + 85)

  // Mostrar cantidad de landmarks detectados (para debug)
  if (predictionQueue.length > 0) {
    const debugText = `Predicciones en cola: ${predictionQueue.length}`
    canvasCtx.fillText(debugText, x + 200, y + 85)
  }
}
