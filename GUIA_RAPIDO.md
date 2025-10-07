# 🎮 Guia Rápido - Pac-Man Webcam

## ▶️ Iniciando o Projeto

```bash
# 1. Instalar dependências (primeira vez)
yarn install

# 2. Iniciar o servidor de desenvolvimento
yarn dev

# 3. Abrir no navegador
# Acesse: http://localhost:3000
```

## 🎯 Como Usar

### Passo 1: Treinar o Modelo
1. Permita o acesso à webcam quando solicitado
2. Para cada direção (↑ ↓ ← →):
   - **Clique e SEGURE** o botão "Adicionar Amostra"
   - Faça o gesto desejado (ex: mão levantada para cima)
   - Mantenha pressionado até ter ~30 exemplos
3. Clique em **"TREINAR MODELO"**
4. Aguarde o treinamento (alguns segundos)

### Passo 2: Jogar
1. Clique em **"JOGAR"**
2. Faça os gestos que você treinou
3. O Pac-Man vai se mover automaticamente! 🎉

## 💡 Dicas para Melhor Resultado

### ✅ BOM:
- Use gestos bem diferentes para cada direção
- Adicione 40-60 exemplos por direção
- Mantenha a iluminação constante
- Fundo simples e limpo

### ❌ EVITE:
- Gestos muito parecidos
- Poucos exemplos (menos de 20)
- Mudar de lugar durante o treino
- Fundo muito confuso ou com movimento

## 🎨 Sugestões de Gestos

| Direção | Gesto Sugerido |
|---------|----------------|
| ↑ Cima  | Mão levantada acima da cabeça |
| ↓ Baixo | Mão abaixada, apontando para baixo |
| ← Esquerda | Mão apontando para a esquerda |
| → Direita | Mão apontando para a direita |

## ⚙️ Ajustes de Hiperparâmetros

Se o modelo não estiver funcionando bem, tente:

1. **Mais dados**: Adicione mais exemplos (50-100 por direção)
2. **Mais epochs**: Mude de 20 para 40
3. **Learning rate menor**: Troque 0.0001 por 0.00001
4. **Retreinar**: Depois de ajustar, treine novamente

## 🚨 Problemas Comuns

### "Nenhuma webcam encontrada"
→ Verifique se a webcam está conectada e permitiu o acesso

### Modelo confunde as direções
→ Use gestos mais distintos e adicione mais exemplos

### Jogo não aparece
→ Verifique se o arquivo `public/pacman-google.js` existe

### Erro ao treinar
→ Adicione pelo menos 20 exemplos em TODAS as direções

## 🎓 Como Funciona?

```
Webcam → Captura Imagem → MobileNet → Sua Rede Neural → Predição → Pac-Man
         (224x224)         (Features)    (Treina)      (Direção)  (Move!)
```

## 📊 Estrutura de Treinamento

```
1. MobileNet extrai características das imagens (Transfer Learning)
2. Sua rede neural aprende a classificar essas características
3. Durante o jogo, prevê em tempo real a direção
```

## 🔧 Comandos Úteis

```bash
# Desenvolvimento
yarn dev

# Build de produção
yarn build

# Executar build
yarn start

# Limpar cache (se tiver problemas)
rm -rf .next node_modules
yarn install
```

## 🎮 Pronto para Jogar!

Divirta-se controlando o Pac-Man com gestos! 🎉👻

**Desafio**: Tente conseguir a maior pontuação possível usando apenas gestos!

