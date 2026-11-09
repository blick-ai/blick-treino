# Blick IA: segmentação e classificação da saúde de plantas de milho

Parte de visão computacional do projeto Blick, TCC do Instituto Mauá de Tecnologia (grupo CMD04, 2026). O sistema completo tem um robô de coleta georreferenciada, uma API na nuvem e um dashboard. Este repositório reúne apenas os dois modelos de rede neural, os notebooks de treino e os resultados.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blick-ai/blick-treino/blob/main/notebooks/pipeline_milho_executavel.ipynb)

## O que o pipeline faz

Cada foto passa por três estágios:

1. **Segmentação de instâncias** (YOLO26n-seg): encontra cada planta de milho e devolve um polígono por planta.
2. **Isolamento**: cada polígono é usado para recortar a planta e pintar de preto tudo que está fora dela.
3. **Classificação binária** (YOLO26n-cls): cada recorte é classificado como `healthy` ou `unhealthy`, com a confiança.

**Regra por imagem:** basta uma planta `unhealthy` para a foto inteira ser `unhealthy`. Se nenhuma planta for detectada, o veredito é "sem milho detectado".

## Dados

| Conjunto | Treino | Validação | Teste | Total |
|---|---|---|---|---|
| Segmentação (imagens) | 4.988 | 1.222 | 673 | 6.883 |
| Instâncias de milho | 30.883 | 6.050 | 3.795 | 40.728 |
| Imagens sem milho | 496 | 133 | 63 | 692 |
| Classificação (recortes) | 17.649 | 5.054 | 2.517 | 25.220 |
| healthy | 15.905 | 4.533 | 2.267 | 22.705 |
| unhealthy | 1.744 | 521 | 250 | 2.515 |

- **Segmentação:** 2.131 das 6.883 imagens são fotos do campo de milho do IMT (câmera ZED, 1080p). As demais vêm de um dataset público do Roboflow. Anotação por polígonos, uma classe (`corn`).
- **Classificação:** só fotos de campo. Os 25.220 recortes vêm de 2.460 fotos distintas e foram gerados pelo próprio segmentador, com fundo preto.
- **Rótulo `unhealthy`:** plantas com um adesivo fixado na folha simulando sintoma (retângulo ou bolinha, amarelo ou marrom). Plantas sem adesivo são `healthy`.
- Os arquivos de recorte seguem o padrão `<id_da_foto_original>_plant<n>.jpg`.

O dataset não está nas pastas do repositório (tamanho e licença das imagens públicas). Os dois zips estão no [Release `dataset-v1`](https://github.com/blick-ai/blick-treino/releases/tag/dataset-v1): `blick-corn-analyzer.zip` (segmentação, COCO) e `blick-classification.zip` (classificação, Folder Structure).

## Treinamento

| | Segmentador | Classificador |
|---|---|---|
| Modelo | yolo26n-seg (2,69 M parâmetros) | yolo26n-cls (1,53 M parâmetros) |
| Pré-treino | pesos da Ultralytics | pesos da Ultralytics |
| Entrada | 640×640 | 224×224 |
| Épocas (máx.) | 100 | 100 |
| Paciência | 20 | 20 |
| Lote | 16 | 64 |
| Épocas efetivas | 100 | 72 (parada antecipada) |
| GPU | Tesla T4 (Colab) | Tesla T4 (Colab) |

Demais hiperparâmetros no padrão da Ultralytics (lr0 = 0,01, semente 0). O `args.yaml` completo de cada treino está em `resultados/`.

## Resultados (conjunto de teste)

**Segmentador** (limiar de confiança 0,40):

| Métrica | Máscara | Caixa |
|---|---|---|
| Precisão | 0,854 | 0,891 |
| Recall | 0,672 | 0,702 |
| mAP@50 | 0,658 | 0,695 |
| mAP@50-95 | 0,388 | 0,532 |

Matriz de confusão (caixa): 2.687 plantas encontradas, 303 falsos positivos, 1.108 plantas não encontradas.

**Classificador:**

| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| healthy | 0,9829 | 0,9890 | 0,9859 | 2.267 |
| unhealthy | 0,8941 | 0,8440 | 0,8683 | 250 |
| macro | 0,9385 | 0,9165 | 0,9271 | 2.517 |

Acurácia 97,46%, AUC-ROC 0,9756, AP da classe `unhealthy` 0,9252. Matriz de confusão: 2.242 / 25 / 39 / 211.

As curvas de loss (treino vs validação) dos dois modelos estão em `resultados/`.

## Como rodar

**Caminho rápido, sem Google Drive:** clique no botão do Colab no topo deste README (ou abra `notebooks/pipeline_milho_executavel.ipynb` no Colab) e use *Ambiente de execução > Executar tudo*. O notebook baixa este repositório (pesos e fotos de exemplo), roda o pipeline nas fotos de `exemplos/entrada`, mostra as curvas de loss e as matrizes de confusão dos dois modelos e traz, na seção 8, um treino ao vivo do classificador (baixa o dataset do Release e treina 3 épocas, uns minutos com GPU). O treino completo, opcional, fica na seção 9. Funciona em CPU, e com GPU (T4) fica mais rápido.

## Estrutura do repositório

```
notebooks/
  pipeline_milho_executavel.ipynb            inferência + resultados + treino opcional (sem Drive)
  pipeline_milho_completo.ipynb              versão original, usa Google Drive
  treino_colab_milho_segmentacao.ipynb       treino do segmentador
  treino_colab_milho_classificacao.ipynb     treino do classificador
weights/
  segmentacao_best.pt                        pesos do yolo26n-seg
  classificacao_best.pt                      pesos do yolo26n-cls
resultados/                                  curvas, matrizes, args.yaml e relatórios dos dois treinos
exemplos/entrada/                            fotos de exemplo usadas pelo notebook
```

Os notebooks de treino precisam do dataset exportado do Roboflow (formato COCO Segmentation para o segmentador, Folder Structure para o classificador). Dependências: ver `requirements.txt`.

## Limitações

- O rótulo `unhealthy` corresponde à presença de adesivo. O classificador foi avaliado como detector de sintomas simulados, e o reconhecimento de doenças reais não foi testado.
- A divisão em treino, validação e teste do classificador foi estratificada por classe e feita por recorte, e não por foto de origem. Recortes de uma mesma foto aparecem em conjuntos diferentes, o que torna as métricas do classificador otimistas.
- O recall do segmentador (0,672 na máscara) limita o que chega ao classificador: cerca de um terço das plantas anotadas não é segmentado.
- Classificação binária: não distingue doença de praga nem identifica a causa.
- Diversidade limitada de cultivares, estágios e condições de luz.

## Licença

Os modelos foram treinados com a biblioteca Ultralytics, licenciada sob AGPL-3.0. Pesos e código derivados seguem a mesma licença.