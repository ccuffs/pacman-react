// Arquivos de imagem do jogo que nao fazem parte do codigo do projeto. Sao baixados na
// primeira visita e guardados no Cache Storage do navegador (funcionam offline depois).
const CACHE_NAME = 'pacman-assets-v1';

const SPRITE_URL =
  'https://storage.googleapis.com/tfjs-examples/assets/webcam-transfer-learning/pacman-sprite.png';
// Fixado em um commit para o conteudo nunca mudar sem querer.
const IMAGES_URL =
  'https://cdn.jsdelivr.net/gh/tensorflow/tfjs-examples@a4aede36138f18e9be1fa46f2e60f06f00a4e7bc/webcam-transfer-learning/images/';

// `bytes` e o tamanho conhecido de cada arquivo, usado para calcular o percentual.
export const ASSETS = [
  { id: 'sprite', url: SPRITE_URL, type: 'image/png', bytes: 6857 },
  { id: 'joystick', url: `${IMAGES_URL}joystick.png`, type: 'image/png', bytes: 14134 },
  { id: 'button', url: `${IMAGES_URL}button.svg`, type: 'image/svg+xml', bytes: 3483 },
  { id: 'pointer', url: `${IMAGES_URL}pointer.svg`, type: 'image/svg+xml', bytes: 451 },
];

async function openCache() {
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    // Sem Cache Storage (contexto inseguro ou modo privado): baixa sempre.
    return null;
  }
}

async function readBody(response, type, onBytes) {
  const reader = response.body.getReader();
  const chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    onBytes(value.byteLength);
  }
  return new Blob(chunks, { type });
}

// Retorna { [id]: objectURL }. onProgress(fracao 0-1).
export async function loadAssets(onProgress) {
  const cache = await openCache();
  const totalBytes = ASSETS.reduce((sum, asset) => sum + asset.bytes, 0);
  let receivedBytes = 0;
  const addBytes = (count) => {
    receivedBytes += count;
    onProgress(Math.min(1, receivedBytes / totalBytes));
  };

  const entries = await Promise.all(
    ASSETS.map(async (asset) => {
      let response = cache ? await cache.match(asset.url) : undefined;
      let saving = Promise.resolve();
      if (!response) {
        response = await fetch(asset.url);
        if (!response.ok) throw new Error(`HTTP ${response.status} ao baixar ${asset.url}`);
        if (cache) saving = cache.put(asset.url, response.clone()).catch(() => {});
      }
      const blob = await readBody(response, asset.type, addBytes);
      await saving;
      return [asset.id, URL.createObjectURL(blob)];
    })
  );
  return Object.fromEntries(entries);
}
