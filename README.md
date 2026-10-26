# Webcam Pac-Man - Controle por Rede Neural

Jogo Pac-Man controlado por uma rede neural treinada com imagens da sua webcam usando TensorFlow.js e React/Vite.

## 🎮 Sobre o Projeto

Este projeto é uma migração do jogo Pac-Man controlado por webcam originalmente desenvolvido pelo Google para React/Vite. Ele usa transfer learning com MobileNet para criar um classificador de imagens que permite controlar o Pac-Man com gestos capturados pela câmera.

## 🚀 Como Funciona

1. **Captura de Exemplos**: Usando sua webcam, você captura imagens para cada direção (cima, baixo, esquerda, direita)
2. **Treinamento**: Uma rede neural é treinada usando transfer learning com MobileNet
3. **Jogo**: A rede neural prevê a direção baseada na imagem da webcam em tempo real e controla o Pac-Man

## 📋 Pré-requisitos

- Node.js 18+ ou superior
- Webcam funcional
- Navegador moderno com suporte a WebGL

## 🛠️ Instalação

```bash
# Instalar dependências
yarn install

# ou com npm
npm install
```

## 🎯 Como Usar

1. **Iniciar o servidor de desenvolvimento**:
```bash
yarn dev

# ou com npm
npm run dev
```

2. **Abrir no navegador**:
Acesse `http://localhost:5173`

3. **Treinar o modelo**:
   - Posicione-se em frente à webcam
   - Para cada direção (cima, baixo, esquerda, direita):
     - Clique e segure o botão "Adicionar Amostra"
     - Faça o gesto ou posição que deseja usar para aquela direção
     - Adicione pelo menos 30-50 exemplos por direção
   - Ajuste os hiperparâmetros se necessário:
     - **Learning Rate**: Taxa de aprendizado (padrão: 0.0001)
     - **Batch Size**: Fração do dataset por batch (padrão: 0.4)
     - **Epochs**: Número de épocas de treinamento (padrão: 40)
     - **Hidden Units**: Unidades na camada oculta (padrão: 200)
   - Clique em "TREINAR MODELO"

4. **Jogar**:
   - Após o treinamento, clique em "JOGAR"
   - Faça os gestos que você treinou para controlar o Pac-Man!

## 📁 Estrutura do Projeto

```
pacman-react/
├── index.html                  # Entrada do Vite
├── vite.config.js
├── public/
│   ├── pacman-google.js        # Motor do jogo Pac-Man (legado)
│   └── models/mobilenet/       # MobileNet hospedado localmente
├── src/
│   ├── main.jsx                # Bootstrap do React
│   ├── App.jsx                 # Componente principal
│   ├── App.module.css          # Estilos do componente
│   └── index.css               # Estilos globais + Pac-Man
└── package.json
```

## 🔧 Tecnologias Utilizadas

- **React 19**: Biblioteca de UI
- **Vite**: Bundler e servidor de desenvolvimento
- **TensorFlow.js**: Machine Learning no navegador
- **TensorFlow.js Data**: Captura de webcam
- **MobileNet**: Modelo pré-treinado para transfer learning

## 📝 Arquivos Migrados

Os seguintes arquivos foram migrados do projeto original:

- `controller_dataset.js` → Integrado em `src/App.jsx` como classe
- `index.js` → Lógica integrada em `src/App.jsx` com hooks React
- `ui.js` → Interface integrada em `src/App.jsx` como componente React
- `index.html` → Estrutura convertida para JSX em `src/App.jsx`
- `pacman-google.js` → Copiado para `public/pacman-google.js`
- Estilos → Migrados para `App.module.css` e `index.css`

## 🎨 Personalizações

### Ajustar Hiperparâmetros

Você pode ajustar os hiperparâmetros através da interface:

- **Learning Rate**: Controla a velocidade de aprendizado
- **Batch Size**: Tamanho do lote de treinamento
- **Epochs**: Quantas vezes o modelo verá todo o dataset
- **Hidden Units**: Complexidade da camada oculta

### Melhorar a Precisão

Para melhorar a precisão do modelo:

1. Adicione mais exemplos (50-100 por direção)
2. Use gestos distintos para cada direção
3. Mantenha iluminação consistente
4. Evite fundos muito complexos
5. Aumente o número de epochs (mas cuidado com overfitting)

## 🐛 Troubleshooting

### "Nenhuma webcam encontrada"
- Verifique se sua webcam está conectada e funcionando
- Permita o acesso à webcam quando solicitado pelo navegador
- Teste em outro navegador se o problema persistir

### Modelo não funciona bem
- Adicione mais exemplos de treinamento
- Use gestos mais distintos entre as direções
- Retreine o modelo com diferentes hiperparâmetros

### Jogo não carrega
- Verifique se o arquivo `pacman-google.js` está em `public/`
- Verifique o console do navegador para erros
- Certifique-se de que todas as dependências foram instaladas

## 📄 Licença

Este projeto é baseado no código original do Google (Apache License 2.0) e foi migrado para React/Vite.
Veja o arquivo [LICENSE](LICENSE).

### Projeto educacional

Este é um projeto **educacional e sem fins lucrativos**, desenvolvido na UFFS. Ele não é afiliado,
patrocinado ou endossado pela BANDAI NAMCO Entertainment Inc. nem pelo Google.

PAC-MAN™ © BANDAI NAMCO Entertainment Inc. O nome, os personagens e os sprites pertencem aos seus
respectivos titulares e não são cobertos pela licença deste repositório.

### Componentes de terceiros

| Componente | Origem | Licença |
| --- | --- | --- |
| Motor do jogo (`public/pacman-google.js`, `public/pacman-sprite.png`) | Google Doodle de 2010 (versão de 2018) | Apache 2.0; PAC-MAN™ © BANDAI NAMCO |
| MobileNet v1 0.25 (`public/models/mobilenet/`) | [tfjs-models](https://github.com/tensorflow/tfjs-models) | Apache 2.0 (veja o [NOTICE](public/models/mobilenet/NOTICE.md)) |
| TensorFlow.js | [tensorflow/tfjs](https://github.com/tensorflow/tfjs) | Apache 2.0 |
| React, Vite, MUI | Respectivos projetos | MIT |

## 🙏 Créditos

- Código original do jogo: Google LLC
- PAC-MAN: BANDAI NAMCO Entertainment Inc.
- Modelo MobileNet: The TensorFlow Authors / Google LLC
- Migração para React/Vite: Este projeto

## 🤝 Contribuindo

Sinta-se à vontade para abrir issues e pull requests com melhorias!
