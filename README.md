source venv/bin/activate

# Image Classifier

Este projeto usa TensorFlow para treinar uma IA capaz de classificar imagens usando o conjunto de dados CIFAR-10 ou imagens personalizadas.

## Como usar

### 1. Instalar as dependências
```bash
pip install -r requirements.txt
```


### 2. Treinar o modelo
```bash
python src/train.py
```


### 3. Fazer previsões
Coloque sua imagem na pasta data/custom_images e execute:

```bash
python src/predict.py
```



---

### Próximos Passos

1. **Testar:** Rodar o treinamento e tente fazer uma previsão com uma imagem personalizada.
2. **Melhorar o Modelo:** Adicionar mais camadas, ajustar hiperparâmetros, talvez usar **transfer learning** para melhorar a precisão.
3. **Deploy:**  Ciar uma API ou um aplicativo que utilize o modelo treinado para fazer classificações de forma automatizada.
