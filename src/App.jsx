import { useEffect, useRef, useState } from 'react';
import styles from './App.module.css';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const GAME_WIDTH = 554;
const GAME_HEIGHT = 184;
const MAX_GAME_SCALE = 2.3;
const NUM_CLASSES = 4;
const CONTROLS = ['up', 'down', 'left', 'right'];
const CONTROL_LABELS = ['cima', 'baixo', 'esquerda', 'direita'];
const CONTROL_CODES = [38, 40, 37, 39];
const PREDICTIONS_PER_SECOND = 15;
const MOBILENET_URLS = [
  '/models/mobilenet/model.json',
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json',
];

// O TensorFlow.js e pesado: so e baixado depois que a pagina e o jogo ja foram exibidos.
let tfPromise;
const loadTf = () => (tfPromise ??= import('@tensorflow/tfjs'));

// O motor do Pac-Man e um script legado que precisa do #logo no DOM; e injetado uma unica vez.
let gameScriptPromise;
function loadGameScript() {
  gameScriptPromise ??= new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = '/pacman-google.js';
    script.async = true;
    script.onload = resolve;
    script.onerror = reject;
    document.body.appendChild(script);
  });
  return gameScriptPromise;
}

function pacman() {
  return window.google && window.google.pacman;
}

function tagError(kind, promise) {
  return promise.catch((cause) => {
    const error = new Error(`${kind}: ${cause?.message ?? cause}`);
    error.kind = kind;
    error.cause = cause;
    throw error;
  });
}

async function loadTruncatedMobileNet(tf) {
  let mobilenet;
  let lastError;
  for (const url of MOBILENET_URLS) {
    try {
      mobilenet = await tf.loadLayersModel(url);
      break;
    } catch (error) {
      lastError = error;
    }
  }
  if (!mobilenet) throw lastError;
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}

class ControllerDataset {
  constructor(tf, numClasses) {
    this.tf = tf;
    this.numClasses = numClasses;
    this.xs = null;
    this.ys = null;
  }

  // Assume a posse do tensor `example`.
  addExample(example, label) {
    const { tf } = this;
    const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label], 'int32'), this.numClasses));

    if (this.xs == null) {
      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
      return;
    }

    const oldX = this.xs;
    const oldY = this.ys;
    this.xs = tf.keep(oldX.concat(example, 0));
    this.ys = tf.keep(oldY.concat(y, 0));
    oldX.dispose();
    oldY.dispose();
    example.dispose();
    y.dispose();
  }

  clear() {
    this.xs?.dispose();
    this.ys?.dispose();
    this.xs = null;
    this.ys = null;
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function Dropdown({ id, label, value, options, onChange, disabled }) {
  return (
    <div className={styles.dropdown}>
      <label htmlFor={id}>{label}</label>
      <div className={styles.select}>
        <select id={id} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

function SamplePad({ direction, label, total, canvasRef, onStart, onStop }) {
  const isActivationKey = (e) => e.key === ' ' || e.key === 'Enter';
  return (
    <div className={styles.thumbBox}>
      <div className={styles.thumbBoxOuter}>
        <div className={styles.thumbBoxInner}>
          <canvas className={styles.thumb} width="224" height="224" ref={canvasRef}></canvas>
        </div>
        <button
          type="button"
          className={styles.recordButton}
          aria-label={`Segure para adicionar amostras para ${label}`}
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture?.(e.pointerId);
            onStart();
          }}
          onPointerUp={onStop}
          onPointerCancel={onStop}
          onLostPointerCapture={onStop}
          onKeyDown={(e) => {
            if (isActivationKey(e)) {
              e.preventDefault();
              if (!e.repeat) onStart();
            }
          }}
          onKeyUp={(e) => isActivationKey(e) && onStop()}
          onBlur={onStop}
          onContextMenu={(e) => e.preventDefault()}
        >
          <span>Segure para gravar</span>
        </button>
      </div>
      <p>
        <span id={`${direction}-total`}>{total}</span> exemplos
      </p>
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState('Carregando MobileNet e webcam...');
  const [totals, setTotals] = useState([0, 0, 0, 0]);
  const [trainStatus, setTrainStatus] = useState('TREINAR MODELO');
  const [isPredicting, setIsPredicting] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [showController, setShowController] = useState(false);
  const [error, setError] = useState(null);
  const [params, setParams] = useState({
    learningRate: '0.0001',
    batchSizeFraction: '0.4',
    epochs: '40',
    denseUnits: '200',
  });
  const [gameScale, setGameScale] = useState(1.5);

  const gameAreaRef = useRef(null);
  const videoRef = useRef(null);
  const tfRef = useRef(null);
  const webcamRef = useRef(null);
  const truncatedMobileNetRef = useRef(null);
  const modelRef = useRef(null);
  const controllerDatasetRef = useRef(null);
  const isPredictingRef = useRef(false);
  const isRecordingRef = useRef(false);

  const thumbCanvasRefs = {
    up: useRef(null),
    down: useRef(null),
    left: useRef(null),
    right: useRef(null),
  };

  const setParam = (name) => (value) => setParams((prev) => ({ ...prev, [name]: value }));

  // Escala o jogo (554x184px nativos) para caber na largura disponivel.
  useEffect(() => {
    const el = gameAreaRef.current;
    if (!el) return;
    const update = () => {
      const available = el.clientWidth - 16;
      setGameScale(Math.max(0.5, Math.min(MAX_GAME_SCALE, available / GAME_WIDTH)));
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Jogo primeiro: o script do Pac-Man carrega logo apos a primeira renderizacao.
  useEffect(() => {
    loadGameScript().catch(console.error);
  }, []);

  // Webcam + MobileNet (em paralelo), depois que o TensorFlow.js for baixado.
  useEffect(() => {
    let mounted = true;

    async function init() {
      try {
        const tf = await loadTf();
        tfRef.current = tf;
        controllerDatasetRef.current = new ControllerDataset(tf, NUM_CLASSES);

        const [webcam, truncatedMobileNet] = await Promise.all([
          tagError(
            'webcam',
            tf.data.webcam(undefined, { resizeWidth: 224, resizeHeight: 224, facingMode: 'user' })
          ),
          tagError('model', loadTruncatedMobileNet(tf)),
        ]);

        if (!mounted) {
          webcam.stop();
          return;
        }
        webcamRef.current = webcam;
        truncatedMobileNetRef.current = truncatedMobileNet;

        // Warm up
        const screenShot = await webcam.capture();
        tf.tidy(() => {
          truncatedMobileNet.predict(screenShot.expandDims(0));
        });
        screenShot.dispose();

        if (mounted) {
          setShowController(true);
          setStatus('');
        }
      } catch (e) {
        console.error(e);
        if (mounted) {
          setStatus('');
          setError({
            kind: e.kind === 'model' ? 'model' : 'webcam',
            message:
              e.kind === 'model'
                ? 'Não foi possível carregar o modelo MobileNet. Verifique a conexão e recarregue a página.'
                : 'Nenhuma webcam encontrada ou permissão negada. Para usar esta demo, use um dispositivo com webcam.',
          });
        }
      }
    }

    init();

    return () => {
      mounted = false;
      isPredictingRef.current = false;
      isRecordingRef.current = false;
      webcamRef.current?.stop();
    };
  }, []);

  // Mostra a previa da webcam assim que o painel de treinamento existe.
  useEffect(() => {
    const stream = webcamRef.current?.stream;
    if (showController && videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [showController]);

  // Capturar imagem da webcam
  async function getImage() {
    const tf = tfRef.current;
    const img = await webcamRef.current.capture();
    const processedImg = tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
    img.dispose();
    return processedImg;
  }

  // Desenhar thumbnail
  async function drawThumb(img, label) {
    const tf = tfRef.current;
    const canvas = thumbCanvasRefs[CONTROLS[label]].current;
    if (!canvas) return;
    const pixels = tf.tidy(() => img.squeeze([0]).add(1).div(2).clipByValue(0, 1));
    await tf.browser.toPixels(pixels, canvas);
    pixels.dispose();
  }

  async function addExample(label) {
    const tf = tfRef.current;
    const img = await getImage();
    const embedding = tf.tidy(() => truncatedMobileNetRef.current.predict(img));
    controllerDatasetRef.current.addExample(embedding, label);
    await drawThumb(img, label);
    img.dispose();
  }

  async function startRecording(label) {
    if (isRecordingRef.current) return;
    isRecordingRef.current = true;
    setError(null);
    try {
      while (isRecordingRef.current) {
        await addExample(label);
        setTotals((prev) => {
          const next = [...prev];
          next[label]++;
          return next;
        });
        await tfRef.current.nextFrame();
      }
    } catch (e) {
      console.error(e);
      setError({ kind: 'train', message: 'Falha ao gravar amostra. Veja o console para detalhes.' });
    } finally {
      isRecordingRef.current = false;
    }
  }

  function stopRecording() {
    isRecordingRef.current = false;
  }

  function clearSamples() {
    controllerDatasetRef.current.clear();
    setTotals([0, 0, 0, 0]);
    setError(null);
    for (const ref of Object.values(thumbCanvasRefs)) {
      const canvas = ref.current;
      canvas?.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  async function train() {
    const tf = tfRef.current;
    const dataset = controllerDatasetRef.current;

    if (totals.some((total) => total === 0)) {
      setError({ kind: 'train', message: 'Adicione amostras para as 4 direções antes de treinar.' });
      return;
    }

    const denseUnits = parseInt(params.denseUnits, 10);
    const learningRate = parseFloat(params.learningRate);
    const epochs = parseInt(params.epochs, 10);
    const batchSize = Math.floor(dataset.xs.shape[0] * parseFloat(params.batchSizeFraction));

    if (!(batchSize > 0)) {
      setError({
        kind: 'train',
        message: 'Tamanho do batch é 0 ou NaN. Adicione mais amostras ou escolha uma fração maior.',
      });
      return;
    }

    setError(null);
    setIsTraining(true);
    setTrainStatus('Treinando...');

    modelRef.current?.dispose();
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
    modelRef.current.compile({ optimizer, loss: 'categoricalCrossentropy' });

    try {
      await modelRef.current.fit(dataset.xs, dataset.ys, {
        batchSize,
        epochs,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            setTrainStatus(`Época ${epoch + 1}/${epochs} · loss ${logs.loss.toFixed(4)}`);
          },
        },
      });
      setModelReady(true);
    } catch (e) {
      console.error(e);
      setError({ kind: 'train', message: 'Falha durante o treinamento. Veja o console para detalhes.' });
    } finally {
      optimizer.dispose();
      setTrainStatus('TREINAR MODELO');
      setIsTraining(false);
    }
  }

  // Predicao continua, limitada a PREDICTIONS_PER_SECOND para nao competir com o jogo.
  async function predict() {
    const tf = tfRef.current;
    const minInterval = 1000 / PREDICTIONS_PER_SECOND;

    while (isPredictingRef.current) {
      const startedAt = performance.now();
      const img = await getImage();
      const predictedClass = tf.tidy(() =>
        modelRef.current.predict(truncatedMobileNetRef.current.predict(img)).as1D().argMax()
      );
      img.dispose();
      const classId = (await predictedClass.data())[0];
      predictedClass.dispose();

      pacman()?.keyPressed(CONTROL_CODES[classId]);

      await tf.nextFrame();
      const wait = minInterval - (performance.now() - startedAt);
      if (wait > 0) await sleep(wait);
    }
  }

  function togglePredict() {
    if (isPredictingRef.current) {
      isPredictingRef.current = false;
      setIsPredicting(false);
      return;
    }
    isPredictingRef.current = true;
    setIsPredicting(true);
    restartGame();
    predict();
  }

  function restartGame() {
    try {
      pacman()?.startGameplay();
    } catch (e) {
      console.error(e);
    }
  }

  const samplePad = (index) => (
    <SamplePad
      direction={CONTROLS[index]}
      label={CONTROL_LABELS[index]}
      total={totals[index]}
      canvasRef={thumbCanvasRefs[CONTROLS[index]]}
      onStart={() => startRecording(index)}
      onStop={stopRecording}
    />
  );

  return (
    <>
      <header className={styles.header}>
        <div className={styles.headerContent}>
          <div className={styles.headerText}>
            Jogue PacMan usando uma <b>INTELIGÊNCIA ARTIFICIAL</b> e a câmera para controlar o jogo.
          </div>
          <img
            src="/cc-logo-black-on-bg-transparent.png"
            alt="CC Logo"
            width="255"
            height="60"
            className={styles.headerLogo}
          />
        </div>
      </header>

      <div className={styles.mainContent}>
        <div className={styles.accordionContainer}>
          {error && error.kind !== 'train' && (
            <div className={styles.noWebcam} role="alert">
              {error.message}
            </div>
          )}

          {status && (
            <div className={styles.status} role="status" aria-live="polite">
              {status}
            </div>
          )}

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
                        Segure o botão para adicionar a <br />
                        visualização atual da <br />
                        câmera como exemplos <br />
                        para esse controle
                      </p>
                    </div>
                    <div className={`${styles.panelCell} ${styles.panelCellCenter}`}>{samplePad(0)}</div>
                    <div className={`${styles.panelCell} ${styles.panelCellRight} ${styles.panelCellFill}`}></div>
                  </div>

                  <div className={styles.panelRowMiddle}>
                    <div className={`${styles.panelCell} ${styles.panelCellLeft}`}>{samplePad(2)}</div>
                    <div className={`${styles.panelCell} ${styles.panelCellCenter} ${styles.panelCellFill}`}>
                      <img height="108" width="129" src="/joystick.png" alt="Joystick" />
                    </div>
                    <div className={`${styles.panelCell} ${styles.panelCellRight}`}>{samplePad(3)}</div>
                  </div>

                  <div className={styles.panelRowBottom}>
                    <div className={`${styles.panelCell} ${styles.panelCellLeft} ${styles.panelCellFill}`}></div>
                    <div className={`${styles.panelCell} ${styles.panelCellCenter}`}>{samplePad(1)}</div>
                    <div className={`${styles.panelCell} ${styles.panelCellRight} ${styles.panelCellFill}`}></div>
                  </div>

                  <div className={styles.actionsRow}>
                    <button
                      type="button"
                      className={styles.secondaryButton}
                      onClick={clearSamples}
                      disabled={isTraining || isPredicting || totals.every((total) => total === 0)}
                    >
                      Limpar amostras
                    </button>
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
                <h2 className={styles.accordionTitle}>Painel de Treinamento</h2>
              </AccordionSummary>
              <AccordionDetails>
                <div className={styles.trainingPanel}>
                  {error && error.kind === 'train' && (
                    <div className={styles.inlineError} role="alert">
                      {error.message}
                    </div>
                  )}

                  <div className={styles.bigButtons}>
                    <button id="train" type="button" onClick={train} disabled={isTraining || isPredicting}>
                      <img width="66" height="66" src="/button.svg" alt="" />
                      <span id="train-status" aria-live="polite">
                        {trainStatus}
                      </span>
                    </button>
                    <button id="predict" type="button" onClick={togglePredict} disabled={!modelReady || isTraining}>
                      <img width="66" height="66" src="/button.svg" alt="" />
                      <span>{isPredicting ? 'PARAR' : 'JOGAR'}</span>
                    </button>
                  </div>

                  <div className={styles.paramsWebcamRow}>
                    <div className={styles.hyperParams}>
                      <Dropdown
                        id="learningRate"
                        label="Learning rate"
                        value={params.learningRate}
                        options={['0.00001', '0.0001', '0.001', '0.003']}
                        onChange={setParam('learningRate')}
                        disabled={isTraining}
                      />
                      <Dropdown
                        id="batchSizeFraction"
                        label="Batch size"
                        value={params.batchSizeFraction}
                        options={['0.05', '0.1', '0.4', '1']}
                        onChange={setParam('batchSizeFraction')}
                        disabled={isTraining}
                      />
                      <Dropdown
                        id="epochs"
                        label="Epochs"
                        value={params.epochs}
                        options={['10', '20', '40']}
                        onChange={setParam('epochs')}
                        disabled={isTraining}
                      />
                      <Dropdown
                        id="dense-units"
                        label="Hidden units"
                        value={params.denseUnits}
                        options={['10', '100', '200']}
                        onChange={setParam('denseUnits')}
                        disabled={isTraining}
                      />
                    </div>

                    <div className={styles.webcamBoxOuter}>
                      <div className={styles.webcamBoxInner}>
                        <video ref={videoRef} autoPlay playsInline muted id="webcam" width="224" height="224"></video>
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
              <div id="pacman-container" className={styles.pacmanContainer} ref={gameAreaRef}>
                <div
                  className={styles.gameStage}
                  style={{ width: GAME_WIDTH * gameScale, height: GAME_HEIGHT * gameScale }}
                >
                  <div id="logo" style={{ transform: `scale(${gameScale})` }}>
                    <div id="logo-l">
                      <div id="logo-b"></div>
                    </div>
                  </div>
                </div>
              </div>
              <div className={styles.actionsRow}>
                <button type="button" className={styles.secondaryButton} onClick={restartGame}>
                  Reiniciar jogo
                </button>
                <span className={styles.hint}>Use as setas do teclado, toque/deslize ou a câmera.</span>
              </div>
            </AccordionDetails>
          </Accordion>
        </div>
      </div>

      <p className={styles.copyright} id="copyright">
        PAC-MAN™ © BANDAI NAMCO Entertainment Inc.
      </p>
    </>
  );
}
