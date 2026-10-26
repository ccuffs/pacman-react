// Etapas mostradas na tela de carregamento. O peso define quanto cada uma conta na barra geral.
export const STEP_DEFS = [
  { id: 'libs', label: 'Biblioteca de IA e motor do jogo', weight: 8 },
  { id: 'assets', label: 'Imagens do jogo', weight: 4 },
  { id: 'download', label: 'Baixar o modelo de IA', weight: 40 },
  { id: 'tfload', label: 'Carregar o modelo no TensorFlow', weight: 10 },
  { id: 'camera', label: 'Acessar a câmera', weight: 8 },
  { id: 'warmup', label: 'Preparar a GPU (WebGL) e compilar o modelo', weight: 30 },
];

export function makeSteps() {
  return Object.fromEntries(
    STEP_DEFS.map((def) => [def.id, { state: 'pending', fraction: 0, detail: '', startedAt: null }])
  );
}

// Percentual geral (0-100). Etapas puladas (ex.: modelo ja salvo) saem do calculo.
export function overallPercent(steps) {
  let total = 0;
  let done = 0;
  for (const def of STEP_DEFS) {
    const step = steps[def.id];
    if (step.state === 'skipped') continue;
    total += def.weight;
    done += def.weight * (step.state === 'done' ? 1 : Math.min(1, step.fraction));
  }
  return total === 0 ? 0 : Math.round((done / total) * 100);
}
