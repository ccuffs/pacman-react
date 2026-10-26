import LinearProgress from '@mui/material/LinearProgress';
import styles from './App.module.css';

function formatSeconds(totalSeconds) {
  const seconds = Math.max(0, Math.round(totalSeconds));
  if (seconds < 60) return `${seconds} s`;
  return `${Math.floor(seconds / 60)} min ${String(seconds % 60).padStart(2, '0')} s`;
}

export default function TrainingProgress({ training }) {
  if (!training) return null;

  if (training.phase === 'done') {
    return (
      <div className={styles.trainingProgress} role="status" aria-live="polite">
        <p className={styles.trainingSummary}>
          Treino concluído em {formatSeconds(training.elapsed)} · loss final {training.loss?.toFixed(4) ?? '-'}
        </p>
      </div>
    );
  }

  if (training.phase === 'preparing') {
    return (
      <div className={styles.trainingProgress} role="status" aria-live="polite">
        <LinearProgress aria-label="Preparando o treino" className={styles.trainingBar} />
        <p className={styles.trainingText}>Preparando o treino (montando o modelo e compilando na GPU)...</p>
      </div>
    );
  }

  const percent = Math.round(training.fraction * 100);
  const eta = training.etaSeconds == null ? 'calculando tempo restante...' : `~${formatSeconds(training.etaSeconds)} restantes`;
  return (
    <div className={styles.trainingProgress}>
      <LinearProgress
        variant="determinate"
        value={percent}
        aria-label="Progresso do treinamento"
        className={styles.trainingBar}
      />
      <p className={styles.trainingText} role="status" aria-live="off">
        Época {Math.min(training.epoch + 1, training.epochs)}/{training.epochs} · {percent}% · {eta}
        {training.loss != null ? ` · loss ${training.loss.toFixed(4)}` : ''}
      </p>
    </div>
  );
}
