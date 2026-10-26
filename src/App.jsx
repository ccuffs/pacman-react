import { useEffect, useRef, useState } from 'react';
import styles from './App.module.css';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import LoadingScreen from './LoadingScreen.jsx';
import TrainingProgress from './TrainingProgress.jsx';
import { makeSteps } from './loadingSteps.js';
import { loadAssets } from './assets.js';

const GAME_WIDTH = 554;
const GAME_HEIGHT = 184;
const MAX_GAME_SCALE = 2.3;
const NUM_CLASSES = 4;
const CONTROLS = ['up', 'down', 'left', 'right'];
const CONTROL_LABELS = ['cima', 'baixo', 'esquerda', 'direita'];
const CONTROL_CODES = [38, 40, 37, 39];
const PREDICTIONS_PER_SECOND = 15;
const MOBILENET_REMOTE_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
// Depois do primeiro download o modelo fica no IndexedDB do navegador (funciona offline).
const MOBILENET_CACHE_URL = 'indexeddb://mobilenet-v1-025-224';

// O TensorFlow.js e pesado: so e baixado depois que a pagina e o jogo ja foram exibidos.
let tfPromise;
const loadTf = () => (tfPromise ??= import('@tensorflow/tfjs'));

// O motor do Pac-Man e um script legado que precisa do #logo no DOM; e injetado uma unica vez.
let gameScriptPromise;
function loadGameScript(getSpriteUrl) {
  gameScriptPromise ??= getSpriteUrl()
    .then(
      (spriteUrl) =>
        new Promise((resolve, reject) => {
          // O motor le a URL do sprite deste global ao ser carregado.
          window.PACMAN_SPRITE_URL = spriteUrl;
          const script = document.createElement('script');
          script.src = '/pacman-google.js';
          script.async = true;
          script.onload = resolve;
          script.onerror = reject;
          document.body.appendChild(script);
        })
    )
    .catch((error) => {
      gameScriptPromise = undefined;
      throw error;
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

function concatBuffers(chunks) {
  const total = chunks.reduce((n, chunk) => n + chunk.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

// Baixa os pesos contando os bytes recebidos, para que a porcentagem seja real
// (o onProgress do tf.loadLayersModel salta de 0 direto para o fim).
// Retorna { manifest, weightSpecs, weightData }.
async function downloadMobileNet(onFraction) {
  const baseUrl = MOBILENET_REMOTE_URL.slice(0, MOBILENET_REMOTE_URL.lastIndexOf('/') + 1);
  const response = await fetch(MOBILENET_REMOTE_URL);
  if (!response.ok) throw new Error(`HTTP ${response.status} ao baixar o modelo`);
  const manifest = await response.json();

  const groups = manifest.weightsManifest;
  const weightSpecs = groups.flatMap((group) => group.weights);
  const paths = groups.flatMap((group) => group.paths);
  // Os pesos do MobileNet sao todos float32 (4 bytes), sem quantizacao.
  const totalBytes = weightSpecs.reduce((n, spec) => n + 4 * spec.shape.reduce((a, b) => a * b, 1), 0);

  let receivedBytes = 0;
  const shards = await Promise.all(
    paths.map(async (path) => {
      const res = await fetch(baseUrl + path);
      if (!res.ok) throw new Error(`HTTP ${res.status} ao baixar ${path}`);
      const reader = res.body.getReader();
      const chunks = [];
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        receivedBytes += value.byteLength;
        onFraction(Math.min(1, receivedBytes / totalBytes), receivedBytes, totalBytes);
      }
      return concatBuffers(chunks);
    })
  );

  return { manifest, weightSpecs, weightData: concatBuffers(shards).buffer };
}

const formatMB = (bytes) => (bytes / 1024 / 1024).toFixed(1).replace('.', ',');
const nextPaint = () => new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));

// onStep(id, patch) atualiza as etapas "download" e "tfload" da tela de carregamento.
async function loadTruncatedMobileNet(tf, onStep) {
  const cached = Boolean((await tf.io.listModels())[MOBILENET_CACHE_URL]);
  let mobilenet;

  if (cached) {
    onStep('download', { state: 'skipped', detail: 'Modelo salvo neste navegador' });
    onStep('tfload', { state: 'active', fraction: 0.1, detail: 'Iniciando o WebGL...', startedAt: Date.now() });
    await tf.ready();
    await nextPaint();
    onStep('tfload', { fraction: 0.4, detail: 'Lendo o modelo salvo...' });
    mobilenet = await tf.loadLayersModel(MOBILENET_CACHE_URL);
  } else {
    // Primeira vez (ou cache limpo): baixa do Google e guarda no navegador.
    onStep('download', { state: 'active', fraction: 0, detail: 'Conectando...', startedAt: Date.now() });
    const { manifest, weightSpecs, weightData } = await downloadMobileNet((fraction, received, total) =>
      onStep('download', { fraction, detail: `${formatMB(received)} de ${formatMB(total)} MB` })
    );
    onStep('download', { state: 'done', fraction: 1, detail: '' });

    onStep('tfload', { state: 'active', fraction: 0.1, detail: 'Iniciando o WebGL...', startedAt: Date.now() });
    await tf.ready();
    await nextPaint();
    onStep('tfload', { fraction: 0.4, detail: 'Montando as camadas do modelo...' });
    mobilenet = await tf.loadLayersModel(
      tf.io.fromMemory({ modelTopology: manifest.modelTopology, weightSpecs, weightData })
    );
    onStep('tfload', { fraction: 0.7, detail: 'Salvando no navegador para uso offline...' });
    try {
      await mobilenet.save(MOBILENET_CACHE_URL);
    } catch (e) {
      console.warn('Nao foi possivel guardar o modelo para uso offline:', e);
    }
  }

  onStep('tfload', { fraction: 0.9, detail: 'Recortando as camadas usadas...' });
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  const truncated = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
  onStep('tfload', { state: 'done', fraction: 1, detail: '' });
  return truncated;
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
  const [steps, setSteps] = useState(makeSteps);
  const [attempt, setAttempt] = useState(0);
  const [assets, setAssets] = useState({});
  const [totals, setTotals] = useState([0, 0, 0, 0]);
  const [training, setTraining] = useState(null);
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

  // Biblioteca de IA, jogo, modelo e webcam, com progresso por etapa para a tela de carregamento.
  useEffect(() => {
    let mounted = true;

    // Atualiza uma etapa; um patch com `state` troca o estado, senao so a fracao/detalhe.
    const step = (id, patch) => {
      if (!mounted) return;
      setSteps((prev) => ({ ...prev, [id]: { ...prev[id], ...patch } }));
    };
    const start = (id, detail) => step(id, { state: 'active', fraction: 0, detail, startedAt: Date.now() });
    const finish = (id) => step(id, { state: 'done', fraction: 1, detail: '' });

    async function init() {
      setError(null);
      setSteps(makeSteps());
      try {
        start('libs', 'Baixando o TensorFlow.js e o motor do Pac-Man...');
        start('assets', 'Baixando sprite, joystick e botões...');
        let libsDone = 0;
        const libStep = (promise) =>
          promise.then((value) => {
            step('libs', { fraction: ++libsDone / 2 });
            return value;
          });
        const assetsPromise = loadAssets((fraction) => step('assets', { fraction }));
        const [tf, loadedAssets] = await Promise.all([
          libStep(loadTf()),
          assetsPromise.then((urls) => {
            finish('assets');
            return urls;
          }),
          libStep(loadGameScript(() => assetsPromise.then((urls) => urls.sprite))),
        ]);
        finish('libs');
        document.documentElement.style.setProperty('--pointer-url', `url("${loadedAssets.pointer}")`);
        if (mounted) setAssets(loadedAssets);
        tfRef.current = tf;
        controllerDatasetRef.current?.clear();
        controllerDatasetRef.current = new ControllerDataset(tf, NUM_CLASSES);

        start('camera', 'Aguardando a permissão da câmera...');
        const [cameraResult, modelResult] = await Promise.allSettled([
          tagError(
            'webcam',
            tf.data.webcam(undefined, { resizeWidth: 224, resizeHeight: 224, facingMode: 'user' })
          ).then((webcam) => {
            finish('camera');
            return webcam;
          }),
          tagError('model', loadTruncatedMobileNet(tf, step)),
        ]);

        const webcam = cameraResult.status === 'fulfilled' ? cameraResult.value : null;
        if (!mounted || modelResult.status === 'rejected') {
          webcam?.stop();
          if (!mounted) return;
          throw modelResult.reason;
        }
        if (!webcam) throw cameraResult.reason;

        webcamRef.current = webcam;
        const truncatedMobileNet = modelResult.value;
        truncatedMobileNetRef.current = truncatedMobileNet;

        // Warm up: a primeira inferencia compila os shaders na GPU e e a parte mais lenta.
        start('warmup', 'Capturando uma imagem da câmera...');
        await nextPaint();
        const screenShot = await webcam.capture();
        step('warmup', { fraction: 0.3, detail: 'Compilando o modelo na GPU (pode levar alguns segundos)...' });
        await nextPaint();
        const warmupOutput = tf.tidy(() => truncatedMobileNet.predict(screenShot.expandDims(0)));
        await warmupOutput.data();
        warmupOutput.dispose();
        screenShot.dispose();
        finish('warmup');

        if (mounted) {
          setShowController(true);
        }
      } catch (e) {
        console.error(e);
        if (mounted) {
          setError({
            kind: e.kind === 'webcam' ? 'webcam' : 'model',
            message:
              e.kind === 'webcam'
                ? 'Nenhuma webcam encontrada ou permissão negada. Para usar esta demo, use um dispositivo com webcam.'
                : 'Não foi possível baixar os arquivos do jogo (modelo de IA e imagens). Na primeira vez é preciso estar conectado à internet. Conecte-se e tente novamente.',
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
      webcamRef.current = null;
    };
  }, [attempt]);

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
    setTraining((prev) => (prev?.phase === 'done' ? null : prev));
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
    setTraining(null);
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
    setTraining({ phase: 'preparing', epochs });
    // Deixa a barra "preparando" aparecer antes do trabalho pesado (montar e compilar o modelo).
    await nextPaint();

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

    const batchesPerEpoch = Math.ceil(dataset.xs.shape[0] / batchSize);
    const startedAt = performance.now();
    const epochEnds = [];
    let currentEpoch = 0;
    let lastLoss = null; // loss da ultima epoca (a do batch oscila demais)
    let lastRender = 0;

    // Tempo restante: media das epocas depois da primeira (a primeira inclui a compilacao dos shaders).
    const estimateEta = (progressInEpochs) => {
      if (epochEnds.length < 2) return null;
      const averageEpoch = (epochEnds[epochEnds.length - 1] - epochEnds[0]) / (epochEnds.length - 1);
      return (Math.max(0, epochs - progressInEpochs) * averageEpoch) / 1000;
    };

    try {
      await modelRef.current.fit(dataset.xs, dataset.ys, {
        batchSize,
        epochs,
        callbacks: {
          onEpochBegin: (epoch) => {
            currentEpoch = epoch;
          },
          onBatchEnd: (batch) => {
            const now = performance.now();
            if (now - lastRender < 100) return;
            lastRender = now;
            const progressInEpochs = currentEpoch + Math.min(1, (batch + 1) / batchesPerEpoch);
            setTraining({
              phase: 'running',
              epoch: currentEpoch,
              epochs,
              fraction: progressInEpochs / epochs,
              etaSeconds: estimateEta(progressInEpochs),
              loss: lastLoss,
            });
          },
          onEpochEnd: (epoch, logs) => {
            epochEnds.push(performance.now());
            lastLoss = logs.loss;
            lastRender = performance.now();
            setTraining({
              phase: 'running',
              epoch: Math.min(epoch + 1, epochs - 1),
              epochs,
              fraction: (epoch + 1) / epochs,
              etaSeconds: estimateEta(epoch + 1),
              loss: lastLoss,
            });
          },
        },
      });
      setModelReady(true);
      setTraining({ phase: 'done', elapsed: (performance.now() - startedAt) / 1000, loss: lastLoss });
    } catch (e) {
      console.error(e);
      setTraining(null);
      setError({ kind: 'train', message: 'Falha durante o treinamento. Veja o console para detalhes.' });
    } finally {
      optimizer.dispose();
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

  const loading = !showController && !(error && error.kind === 'webcam');

  return (
    <>
      {loading && (
        <LoadingScreen
          steps={steps}
          error={error && error.kind === 'model' ? error : null}
          onRetry={() => setAttempt((n) => n + 1)}
        />
      )}

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
          {error && error.kind === 'webcam' && (
            <div className={styles.noWebcam} role="alert">
              {error.message}
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
                      <img height="108" width="129" src={assets.joystick} alt="Joystick" />
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
                      <img width="66" height="66" src={assets.button} alt="" />
                      <span id="train-status">{isTraining ? 'TREINANDO...' : 'TREINAR MODELO'}</span>
                    </button>
                    <button id="predict" type="button" onClick={togglePredict} disabled={!modelReady || isTraining}>
                      <img width="66" height="66" src={assets.button} alt="" />
                      <span>{isPredicting ? 'PARAR' : 'JOGAR'}</span>
                    </button>
                  </div>

                  <TrainingProgress training={training} />

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
