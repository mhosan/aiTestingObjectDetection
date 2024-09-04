const axios = require('axios');
const fs = require('fs');
const path = require('path');

// Configuración del modelo
const MODEL_ID = 'facebook/detr-resnet-50';
const API_URL = `https://api-inference.huggingface.co/models/${MODEL_ID}`;

// Función para cargar la imagen
async function loadImage(imagePath) {
  const imageBuffer = await fs.promises.readFile(imagePath);
  return imageBuffer.toString('base64');
}

// Función para realizar la inferencia
async function detectPiscinas(imageData) {
  try {
    const response = await axios.post(API_URL, {
      inputs: imageData,
      options: {
        wait_for_model: true,
        use_gpu: false, // Usamos CPU ya que no tenemos GPU
        top_k: 50, // Máximo número de detecciones por imagen
      },
    });

    return response.data;
  } catch (error) {
    console.error('Error en la inferencia:', error);
    throw error;
  }
}

// Función principal
async function main() {
  const imagePath = path.join(__dirname, 'input', 'piscina_satelite.jpg');
  const imageData = await loadImage(imagePath);

  try {
    const results = await detectPiscinas(imageData);

    console.log('Detecciones:');
    results.forEach((detection) => {
      console.log(`Objeto: ${detection.label}, Confianza: ${detection.score.toFixed(2)}, Bounding Box: ${JSON.stringify(detection.box)}`);
    });
  } catch (error) {
    console.error('Error:', error.message);
  }
}

main();
