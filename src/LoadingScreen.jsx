import { useEffect, useRef, useState } from 'react';
import LinearProgress from '@mui/material/LinearProgress';
import styles from './App.module.css';
import { STEP_DEFS, overallPercent } from './loadingSteps.js';

const ICONS = { pending: '○', active: '●', done: '✓', skipped: '✓' };

function StepRow({ def, step, now }) {
  const percent = Math.round(Math.min(1, step.fraction) * 100);
  const elapsed = step.startedAt ? Math.floor((now - step.startedAt) / 1000) : 0;
  return (
    <li className={`${styles.stepRow} ${styles[`step_${step.state}`]}`}>
      <span className={styles.stepIcon} aria-hidden="true">
        {ICONS[step.state]}
      </span>
      <div className={styles.stepBody}>
        <div className={styles.stepHeader}>
          <span>{def.label}</span>
          <span className={styles.stepValue}>
            {step.state === 'active' && `${percent}%`}
            {step.state === 'done' && '100%'}
            {step.state === 'skipped' && 'já salvo'}
          </span>
        </div>
        {step.state === 'active' && (
          <>
            <LinearProgress
              variant="determinate"
              value={percent}
              aria-label={def.label}
              className={styles.stepBar}
            />
            <span className={styles.stepDetail}>
              {step.detail}
              {elapsed >= 2 ? ` (${elapsed} s)` : ''}
            </span>
          </>
        )}
        {step.state === 'skipped' && step.detail && <span className={styles.stepDetail}>{step.detail}</span>}
      </div>
    </li>
  );
}

export default function LoadingScreen({ steps, error, onRetry }) {
  // A barra geral nunca recua (etapas puladas mudam o total depois que o carregamento comeca).
  const maxRef = useRef(0);
  maxRef.current = Math.max(maxRef.current, overallPercent(steps));
  const percent = maxRef.current;

  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(timer);
  }, []);

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
            <LinearProgress
              variant="determinate"
              value={percent}
              aria-label="Progresso geral do carregamento"
              className={styles.loadingBar}
            />
            <p className={styles.loadingPercent}>{percent}%</p>

            <ul className={styles.stepList}>
              {STEP_DEFS.map((def) => (
                <StepRow key={def.id} def={def} step={steps[def.id]} now={now} />
              ))}
            </ul>

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
