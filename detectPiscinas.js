import { HfInference } from "@huggingface/inference";
import { config } from 'dotenv';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

config();
const hf = new HfInference(process.env.HF_TOKEN);

// Obtener el directorio actual en ES6
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuración del modelo
const MODEL_ID = 'facebook/detr-resnet-50';

// Función para cargar la imagen
async function loadImage(imagePath) {
  try {
    const imageBuffer = await fs.promises.readFile(imagePath);
    return Buffer.from(imageBuffer).toString('base64');
  } catch (error) {
    console.error(`Error al cargar la imagen: ${error.message}`);
    throw error;
  }
}

// Función para realizar la inferencia
async function detectPiscinas(imageData, fileName) { // Asegúrate de que el nombre del argumento es 'fileName'
  try {
    const response = await hf.objectDetection({
      model: MODEL_ID,
      data: imageData,
      options: {
        wait_for_model: true,
        use_gpu: false,
      },
    });

    return response;
  } catch (error) {
    console.error(`Error en la inferencia para la imagen: ${fileName}`);
    console.error('Detalle del error:', error.message);
    throw error;
  }
}

// Función principal
async function main() {
  const inputDir = path.join(__dirname, 'input');

  try {
    const files = await fs.promises.readdir(inputDir);

    for (const file of files) { // Aquí 'file' se refiere al nombre del archivo actual en la iteración
      const imagePath = path.join(inputDir, file);
      const imageData = await loadImage(imagePath);

      console.log(`Procesando imagen: ${file}`);
      const results = await detectPiscinas(imageData, file); // Pasamos 'file' como 'fileName'

      console.log('Detecciones:');
      results.forEach((detection) => {
        console.log(`Objeto: ${detection.label}, Confianza: ${detection.score.toFixed(2)}, Bounding Box: ${JSON.stringify(detection.box)}`);
      });
    }
  } catch (error) {
    console.error('Error:', error.message);
  }
}

main();

