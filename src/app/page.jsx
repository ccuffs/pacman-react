'use client';

import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';
import styles from './page.module.css';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const NUM_CLASSES = 4;
const CONTROLS = ['up', 'down', 'left', 'right'];
const CONTROL_CODES = [38, 40, 37, 39];

// ControllerDataset class
class ControllerDataset {
  constructor(numClasses) {
    this.numClasses = numClasses;
  }

  addExample(example, label) {
    const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));

    if (this.xs == null) {
      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));

      oldX.dispose();
      oldY.dispose();
      y.dispose();
    }
  }
}

export default function Home() {
  const [status, setStatus] = useState('Carregando MobileNet...');
  const [totals, setTotals] = useState([0, 0, 0, 0]);
  const [trainStatus, setTrainStatus] = useState('TREINAR MODELO');
  const [isPredicting, setIsPredicting] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [showController, setShowController] = useState(false);
  const [noWebcam, setNoWebcam] = useState(false);

  const webcamRef = useRef(null);
  const truncatedMobileNetRef = useRef(null);
  const modelRef = useRef(null);
  const controllerDatasetRef = useRef(new ControllerDataset(NUM_CLASSES));
  const isPredictingRef = useRef(false);
  const mouseDownRef = useRef(false);

  const thumbCanvasRefs = {
    up: useRef(null),
    down: useRef(null),
    left: useRef(null),
    right: useRef(null),
  };

  // Carregar MobileNet truncado
  async function loadTruncatedMobileNet() {
    let mobilenet;
    try {
      // Tentar carregar do arquivo local primeiro
      mobilenet = await tf.loadLayersModel('/model.json');
      console.log('Modelo carregado do arquivo local');
    } catch (error) {
      // Se falhar, carregar da URL remota
      console.log('Carregando modelo da URL remota...');
      mobilenet = await tf.loadLayersModel(
        'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
      );
    }
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
  }

  // Capturar imagem da webcam
  async function getImage() {
    const img = await webcamRef.current.capture();
    const processedImg = tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
    img.dispose();
    return processedImg;
  }

  // Desenhar thumbnail
  function drawThumb(img, label) {
    const thumbCanvas = thumbCanvasRefs[CONTROLS[label]].current;
    if (thumbCanvas) {
      const [width, height] = [224, 224];
      const ctx = thumbCanvas.getContext('2d');
      const imageData = new ImageData(width, height);
      const data = img.dataSync();
      for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
        imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
        imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
        imageData.data[j + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);
    }
  }

  // Handler para adicionar exemplos
  async function addExample(label) {
    const img = await getImage();
    controllerDatasetRef.current.addExample(
      truncatedMobileNetRef.current.predict(img),
      label
    );
    drawThumb(img, label);
    img.dispose();
  }

  // Treinar modelo
  async function train() {
    if (controllerDatasetRef.current.xs == null) {
      alert('Adicione alguns exemplos antes de treinar!');
      return;
    }

    setIsTraining(true);
    setTrainStatus('Treinando...');

    const denseUnits = parseInt(document.getElementById('dense-units').value);
    const learningRate = parseFloat(document.getElementById('learningRate').value);
    const batchSizeFraction = parseFloat(document.getElementById('batchSizeFraction').value);
    const epochs = parseInt(document.getElementById('epochs').value);

    modelRef.current = tf.sequential({
      layers: [
        tf.layers.flatten({
          inputShape: truncatedMobileNetRef.current.outputs[0].shape.slice(1),
        }),
        tf.layers.dense({
          units: denseUnits,
          activation: 'relu',
          kernelInitializer: 'varianceScaling',
          useBias: true,
        }),
        tf.layers.dense({
          units: NUM_CLASSES,
          kernelInitializer: 'varianceScaling',
          useBias: false,
          activation: 'softmax',
        }),
      ],
    });

    const optimizer = tf.train.adam(learningRate);
    modelRef.current.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });

    const batchSize = Math.floor(
      controllerDatasetRef.current.xs.shape[0] * batchSizeFraction
    );

    if (!(batchSize > 0)) {
      alert('Tamanho do batch é 0 ou NaN. Escolha uma fração diferente de zero.');
      setIsTraining(false);
      return;
    }

    await modelRef.current.fit(controllerDatasetRef.current.xs, controllerDatasetRef.current.ys, {
      batchSize,
      epochs,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          setTrainStatus('Loss: ' + logs.loss.toFixed(5));
        },
      },
    });

    setTrainStatus('TREINAR MODELO');
    setIsTraining(false);
  }

  // Predição contínua
  async function predict() {
    while (isPredictingRef.current) {
      const img = await getImage();
      const embeddings = truncatedMobileNetRef.current.predict(img);
      const predictions = modelRef.current.predict(embeddings);
      const predictedClass = predictions.as1D().argMax();
      const classId = (await predictedClass.data())[0];
      img.dispose();

      if (typeof window !== 'undefined' && window.google && window.google.pacman) {
        window.google.pacman.keyPressed(CONTROL_CODES[classId]);
      }

      await tf.nextFrame();
    }
  }

  // Handler para adicionar exemplos ao segurar o botão
  async function handleMouseDown(label) {
    mouseDownRef.current = true;
    while (mouseDownRef.current) {
      await addExample(label);
      setTotals((prev) => {
        const newTotals = [...prev];
        newTotals[label]++;
        return newTotals;
      });
      await tf.nextFrame();
    }
  }

  function handleMouseUp() {
    mouseDownRef.current = false;
  }

  // Inicializar
  useEffect(() => {
    let mounted = true;

    async function init() {
      try {
        if (typeof window !== 'undefined') {
          const videoElement = document.getElementById('webcam');
          const webcam = await tfd.webcam(videoElement, {
            resizeWidth: 224,
            resizeHeight: 224,
            facingMode: 'user'
          });
          webcamRef.current = webcam;

          const truncatedMobileNet = await loadTruncatedMobileNet();
          truncatedMobileNetRef.current = truncatedMobileNet;

          // Warm up
          const screenShot = await webcam.capture();
          truncatedMobileNet.predict(screenShot.expandDims(0));
          screenShot.dispose();

          if (mounted) {
            setShowController(true);
            setStatus('');
          }
        }
      } catch (e) {
        console.error(e);
        if (mounted) {
          setNoWebcam(true);
        }
      }
    }

    init();

    return () => {
      mounted = false;
    };
  }, []);

  // Carregar script do Pac-Man
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const script = document.createElement('script');
      script.src = '/pacman-google.js';
      script.async = true;
      document.body.appendChild(script);

      return () => {
        document.body.removeChild(script);
      };
    }
  }, []);

  return (
    <>
      <header className={styles.header}>
        <div className={styles.headerContent}>
          <div className={styles.headerText}>
            Jogue PacMan usando uma <b>Rede Neural</b> e a câmera para controlar o jogo.
          </div>
          <img
            src="/cc-logo-black-on-bg-transparent.png"
            alt="CC Logo"
            className={styles.headerLogo}
          />
        </div>
      </header>

      {noWebcam && (
        <div className={styles.noWebcam}>
          Nenhuma webcam encontrada. <br />
          Para usar esta demo, use um dispositivo com webcam.
        </div>
      )}

      {status && <div className={styles.status}>{status}</div>}

      <div className={styles.mainContent}>
        <div className={styles.accordionContainer}>
          {showController && (
          <Accordion defaultExpanded>
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              aria-controls="joystick-content"
              id="joystick-header"
            >
              <h2 className={styles.accordionTitle}>Controles de Movimento</h2>
            </AccordionSummary>
            <AccordionDetails>
              <div className={styles.joystickPanel}>
            <div className={styles.panelRowTop}>
              <div className={`${styles.panelCell} ${styles.panelCellLeft} ${styles.panelCellFill}`}>
                <p className={styles.helpText}>
                  Clique para adicionar a <br />
                  visualização atual da <br />
                  câmera como um exemplo <br />
                  para esse controle
                </p>
              </div>

              <div className={`${styles.panelCell} ${styles.panelCellCenter}`}>
                <div className={styles.thumbBox}>
                  <div className={styles.thumbBoxOuter}>
                    <div className={styles.thumbBoxInner}>
                      <canvas className={styles.thumb} width="224" height="224" ref={thumbCanvasRefs.up}></canvas>
                    </div>
                    <button
                      className={styles.recordButton}
                      onMouseDown={() => handleMouseDown(0)}
                      onMouseUp={handleMouseUp}
                      onMouseLeave={handleMouseUp}
                      onTouchStart={() => handleMouseDown(0)}
                      onTouchEnd={handleMouseUp}
                    >
                      <span>Adicionar Amostra</span>
                    </button>
                  </div>
                  <p>
                    <span id="up-total">{totals[0]}</span> exemplos
                  </p>
                </div>
              </div>

              <div className={`${styles.panelCell} ${styles.panelCellRight} ${styles.panelCellFill}`}></div>
            </div>

            <div className={styles.panelRowMiddle}>
              <div className={`${styles.panelCell} ${styles.panelCellLeft}`}>
                <div className={styles.thumbBox}>
                  <div className={styles.thumbBoxOuter}>
                    <div className={styles.thumbBoxInner}>
                      <canvas className={styles.thumb} width="224" height="224" ref={thumbCanvasRefs.left}></canvas>
                    </div>
                    <button
                      className={styles.recordButton}
                      onMouseDown={() => handleMouseDown(2)}
                      onMouseUp={handleMouseUp}
                      onMouseLeave={handleMouseUp}
                      onTouchStart={() => handleMouseDown(2)}
                      onTouchEnd={handleMouseUp}
                    >
                      <span>Adicionar Amostra</span>
                    </button>
                  </div>
                  <p>
                    <span id="left-total">{totals[2]}</span> exemplos
                  </p>
                </div>
              </div>

              <div className={`${styles.panelCell} ${styles.panelCellCenter} ${styles.panelCellFill}`}>
                <img height="108" width="129" src="/joystick.png" alt="Joystick" />
              </div>

              <div className={`${styles.panelCell} ${styles.panelCellRight}`}>
                <div className={styles.thumbBox}>
                  <div className={styles.thumbBoxOuter}>
                    <div className={styles.thumbBoxInner}>
                      <canvas className={styles.thumb} width="224" height="224" ref={thumbCanvasRefs.right}></canvas>
                    </div>
                    <button
                      className={styles.recordButton}
                      onMouseDown={() => handleMouseDown(3)}
                      onMouseUp={handleMouseUp}
                      onMouseLeave={handleMouseUp}
                      onTouchStart={() => handleMouseDown(3)}
                      onTouchEnd={handleMouseUp}
                    >
                      <span>Adicionar Amostra</span>
                    </button>
                  </div>
                  <p>
                    <span id="right-total">{totals[3]}</span> exemplos
                  </p>
                </div>
              </div>
            </div>

            <div className={styles.panelRowBottom}>
              <div className={`${styles.panelCell} ${styles.panelCellLeft} ${styles.panelCellFill}`}></div>

              <div className={`${styles.panelCell} ${styles.panelCellCenter}`}>
                <div className={styles.thumbBox}>
                  <div className={styles.thumbBoxOuter}>
                    <div className={styles.thumbBoxInner}>
                      <canvas className={styles.thumb} width="224" height="224" ref={thumbCanvasRefs.down}></canvas>
                    </div>
                    <button
                      className={styles.recordButton}
                      onMouseDown={() => handleMouseDown(1)}
                      onMouseUp={handleMouseUp}
                      onMouseLeave={handleMouseUp}
                      onTouchStart={() => handleMouseDown(1)}
                      onTouchEnd={handleMouseUp}
                    >
                      <span>Adicionar Amostra</span>
                    </button>
                  </div>
                  <p>
                    <span id="down-total">{totals[1]}</span> exemplos
                  </p>
                </div>
              </div>

              <div className={`${styles.panelCell} ${styles.panelCellRight} ${styles.panelCellFill}`}></div>
            </div>
              </div>
            </AccordionDetails>
          </Accordion>
        )}

        {showController && (
          <Accordion defaultExpanded={!isPredicting}>
            <AccordionSummary
              expandIcon={<ExpandMoreIcon />}
              aria-controls="training-content"
              id="training-header"
            >
              <h2 className={styles.accordionTitle}> Painel de Treinamento</h2>
            </AccordionSummary>
            <AccordionDetails>
              <div className={styles.trainingPanel}>
                {/* Big buttons. */}
                <div className={styles.bigButtons}>
              <button id="train" onClick={train} disabled={isTraining}>
                <img width="66" height="66" src="/button.svg" alt="" />
                <span id="train-status">{trainStatus}</span>
              </button>
              <button
                id="predict"
                onClick={() => {
                  if (isPredictingRef.current) {
                    isPredictingRef.current = false;
                    setIsPredicting(false);
                  } else {
                    isPredictingRef.current = true;
                    setIsPredicting(true);
                    if (typeof window !== 'undefined' && window.google && window.google.pacman) {
                      window.google.pacman.startGameplay();
                    }
                    predict();
                  }
                }}
                disabled={!modelRef.current}
              >
                <img width="66" height="66" src="/button.svg" alt="" />
                <span>{isPredicting ? 'PARAR' : 'JOGAR'}</span>
              </button>
            </div>

            <div className={styles.paramsWebcamRow}>
              {/* Hyper params. */}
              <div className={styles.hyperParams}>
                {/* Learning rate */}
                <div className={styles.dropdown}>
                  <label>Learning rate</label>
                  <div className={styles.select}>
                    <select id="learningRate" defaultValue="0.0001">
                      <option value="0.00001">0.00001</option>
                      <option value="0.0001">0.0001</option>
                      <option value="0.001">0.001</option>
                      <option value="0.003">0.003</option>
                    </select>
                  </div>
                </div>

                {/* Batch size */}
                <div className={styles.dropdown}>
                  <label>Batch size</label>
                  <div className={styles.select}>
                    <select id="batchSizeFraction" defaultValue="0.4">
                      <option value="0.05">0.05</option>
                      <option value="0.1">0.1</option>
                      <option value="0.4">0.4</option>
                      <option value="1">1</option>
                    </select>
                  </div>
                </div>

                {/* Epochs */}
                <div className={styles.dropdown}>
                  <label>Epochs</label>
                  <div className={styles.select}>
                    <select id="epochs" defaultValue="40">
                      <option value="10">10</option>
                      <option value="20">20</option>
                      <option value="40">40</option>
                    </select>
                  </div>
                </div>

                {/* Hidden units */}
                <div className={styles.dropdown}>
                  <label>Hidden units</label>
                  <div className={styles.select}>
                    <select id="dense-units" defaultValue="200">
                      <option value="10">10</option>
                      <option value="100">100</option>
                      <option value="200">200</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className={styles.webcamBoxOuter}>
                <div className={styles.webcamBoxInner}>
                  <video autoPlay playsInline muted id="webcam" width="224" height="224"></video>
                </div>
              </div>
            </div>
              </div>
            </AccordionDetails>
          </Accordion>
        )}

        <Accordion defaultExpanded>
          <AccordionSummary
            expandIcon={<ExpandMoreIcon />}
            aria-controls="game-content"
            id="game-header"
          >
            <h2 className={styles.accordionTitle}>Jogo Pac-Man</h2>
          </AccordionSummary>
          <AccordionDetails>
            <div id="pacman-container" className={styles.pacmanContainer}>
              <div id="logo">
                <div id="logo-l">
                  <div id="logo-b"></div>
                </div>
              </div>
            </div>
          </AccordionDetails>
        </Accordion>
        </div>
      </div>

      <p className={styles.copyright} id="copyright">PAC-MAN™ © BANDAI NAMCO Entertainment Inc.</p>
    </>
  );
}

