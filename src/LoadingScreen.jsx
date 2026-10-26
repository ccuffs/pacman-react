import LinearProgress from '@mui/material/LinearProgress';
import styles from './App.module.css';

export default function LoadingScreen({ progress, error, onRetry }) {
  return (
    <div className={styles.loadingOverlay} role="status" aria-live="polite">
      <div className={styles.loadingCard}>
        <h1 className={styles.loadingTitle}>Preparando o jogo</h1>

        {error ? (
          <>
            <p className={styles.loadingError} role="alert">
              {error.message}
            </p>
            <button type="button" className={styles.secondaryButton} onClick={onRetry}>
              Tentar novamente
            </button>
          </>
        ) : (
          <>
            <p className={styles.loadingLabel}>{progress.label}</p>
            <LinearProgress
              variant="determinate"
              value={progress.value}
              aria-label="Progresso do carregamento"
              className={styles.loadingBar}
            />
            <p className={styles.loadingPercent}>{progress.value}%</p>
            <p className={styles.loadingHint}>
              Na primeira vez o modelo de IA é baixado (cerca de 2 MB) e guardado neste navegador. Nas próximas vezes
              o jogo abre sem internet.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
