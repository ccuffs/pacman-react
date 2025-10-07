# Webcam Pac-Man - Controle por Rede Neural

Jogo Pac-Man controlado por uma rede neural treinada com imagens da sua webcam usando TensorFlow.js e React/Next.js.

## 🎮 Sobre o Projeto

Este projeto é uma migração do jogo Pac-Man controlado por webcam originalmente desenvolvido pelo Google para React/Next.js. Ele usa transfer learning com MobileNet para criar um classificador de imagens que permite controlar o Pac-Man com gestos capturados pela câmera.

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
Acesse `http://localhost:3000`

3. **Treinar o modelo**:
   - Posicione-se em frente à webcam
   - Para cada direção (cima, baixo, esquerda, direita):
     - Clique e segure o botão "Adicionar Amostra"
     - Faça o gesto ou posição que deseja usar para aquela direção
     - Adicione pelo menos 30-50 exemplos por direção
   - Ajuste os hiperparâmetros se necessário:
     - **Learning Rate**: Taxa de aprendizado (padrão: 0.0001)
     - **Batch Size**: Fração do dataset por batch (padrão: 0.4)
     - **Epochs**: Número de épocas de treinamento (padrão: 20)
     - **Hidden Units**: Unidades na camada oculta (padrão: 100)
   - Clique em "TREINAR MODELO"

4. **Jogar**:
   - Após o treinamento, clique em "JOGAR"
   - Faça os gestos que você treinou para controlar o Pac-Man!

## 📁 Estrutura do Projeto

```
pacman-react/
├── public/
│   └── pacman-google.js       # Motor do jogo Pac-Man
├── src/
│   └── app/
│       ├── globals.css         # Estilos globais + Pac-Man
│       ├── layout.js           # Layout do Next.js
│       ├── page.js             # Componente principal
│       └── page.module.css     # Estilos do componente
└── package.json
```

## 🔧 Tecnologias Utilizadas

- **React 19**: Biblioteca de UI
- **Next.js 15**: Framework React
- **TensorFlow.js**: Machine Learning no navegador
- **TensorFlow.js Data**: Captura de webcam
- **MobileNet**: Modelo pré-treinado para transfer learning

## 📝 Arquivos Migrados

Os seguintes arquivos foram migrados do projeto original:

- `controller_dataset.js` → Integrado em `page.js` como classe
- `index.js` → Lógica integrada em `page.js` com hooks React
- `ui.js` → Interface integrada em `page.js` como componente React
- `index.html` → Estrutura convertida para JSX em `page.js`
- `pacman-google.js` → Copiado para `public/pacman-google.js`
- Estilos → Migrados para `page.module.css` e `globals.css`

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

Este projeto é baseado no código original do Google (Apache License 2.0) e foi migrado para React/Next.js.

PAC-MAN™ © BANDAI NAMCO Entertainment Inc.

## 🙏 Créditos

- Código original do jogo: Google LLC
- PAC-MAN: BANDAI NAMCO Entertainment Inc.
- Migração para React/Next.js: Este projeto

## 🤝 Contribuindo

Sinta-se à vontade para abrir issues e pull requests com melhorias!
