# 🌸 Iris Classification Problem com Redes Neurais Artificiais

Este projeto demonstra como construir, treinar e avaliar uma Rede Neural Artificial (RNA) simples usando TensorFlow/Keras para classificar espécies de flores de íris a partir do conjunto de dados clássico de Íris. Inclui visualização de modelos, utilitários de teste e animações opcionais para fins educacionais.

## 📁 Estrutura do Projeto

```
main.py # Script principal para treinamento, avaliação, visualização e teste
README.md # Este arquivo
```

## 🚀 Recursos

- Carrega e pré-processa o conjunto de dados Iris
- Codificação one-hot para rótulos (target) categóricos
- Divide os dados em conjuntos de treinamento e teste
- Treina uma RNA básica com TensorFlow/Keras
- Avalia a precisão do modelo
- Anima projeções de recursos 2D do conjunto de dados
- Testa previsões em amostras aleatórias
- Suporta construção avançada de modelos com:
- Dropout
- Normalização em Lote
- Regularização L2

## 🧠 Arquitetura do Modelo (Básica)

- **Camada de Entrada**: 4 features (variáveis) (comprimento/largura de sépala/pétala)
- **Camada Oculta**: Camada totalmennte conectada (densa) com 10 neurônios e ativação ReLU
- **Camada de Saída**: Camada densa com 3 unidades e função de ativação softmax

## 🛠️ Requisitos

- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- NumPy
- Matplotlib (para animação)

Instale as dependências usando:

```bash
pip3 install tensorflow scikit-learn numpy matplotlib
```

Ou utilize o arquivo requirements.txt
```bash
pip3 install -r requirements.txt
```

## ▶️ Uso

Para executar o treinamento e a avaliação, basta executar o arquivo no python:

```bash
python3 -i main.py
```

Para visualizar a animação do dataset, execute o arquivo em forma interativa (como no bash acima):

```python
animate()
```

Para testar o modelo em amostras random (`n` pode ser passado como parametro):

```python
test_random(n=10)
```

## 🧪 Construtor de Modelo Custom

A função `build_ann_model()` serve para criar uma arquitetura de RNA para teste, com dropout opcional, normalização em lote e regularização L2.

Esta função pode ser manualmente integrada ao ciclo de treino do script, substituindo a declaração de modelo.

### Exemplo:

```python
model = build_ann_model(
input_dim=4,
hidden_layers=[16, 8],
dropout_rate=0.2,
use_batchnorm=True,
l2_reg=0.001
)
```

## 📊 Exemplo de Saída

```plaintext
Número de GPUs Disponíveis: 0

Tempo de treinamento: 7.6345 segundos
Precisão: 0,9999

✅ Previsões corretas: 10/10 | Total: 10
```

## 📌 Observações

- O uso da GPU é desabilitado por padrão (`os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`) para este experimento.
- A função `plot_model()` é comentada pois envolve a instalação de `graphviz` no sistema operacional para um resumo visual do modelo.
- A animação utiliza dimensões de sépala e pétala para uma projeção 2D intuitiva.

## 📚 Informações do Conjunto de Dados

- **Fonte**: [Repositório de Aprendizado de Máquina da UCI](https://archive.ics.uci.edu/ml/datasets/iris)
- **Classes**: Setosa, Versicolor, Virginica
- **Características**: Comprimento da Sépala, Largura da Sépala, Comprimento da Pétala, Largura da Pétala