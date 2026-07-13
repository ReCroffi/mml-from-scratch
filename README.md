# MML from scratch

**🌐 Idioma:** **Português** · [English](README_en.md)

[![Tests](https://github.com/ReCroffi/mml-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/ReCroffi/mml-from-scratch/actions/workflows/tests.yml)

> Implementação do zero (NumPy) de PCA e regressão linear, aplicada a um dataset real —
> encerrando a especialização *Mathematics for Machine Learning* (Imperial College / Coursera).

> ⚠️ **Em construção.** O PCA e a regressão linear (gradiente descendente + equação normal)
> estão implementados, validados e cobertos por testes automatizados (`pytest`). O notebook de
> análise e a comparação com SVD estão nas próximas etapas (ver [Pipeline](#pipeline)).

## Objetivo

Implementar PCA (e, em seguida, regressão linear) **na mão**, em NumPy puro — matriz de
covariância, eigendecomposition, projeção e reconstrução — sem chamar `sklearn.decomposition.PCA`.
A graça não é reduzir dimensionalidade (qualquer biblioteca faz isso em uma linha); é **entender
a matemática por baixo** e provar que a implementação está correta validando cada resultado
contra o sklearn.

## Dataset

**Wine** (`sklearn.datasets.load_wine`): 178 amostras, 13 features numéricas, 3 classes de
vinho (~59 amostras cada — bem balanceado).

Escolhido de propósito pelas **escalas díspares** entre features, o que torna a padronização
não-negociável antes do PCA:

| feature   | desvio padrão | variância (≈ σ²) |
|-----------|--------------:|-----------------:|
| `proline` |         ~315  |          ~99.000 |
| `magnesium` (2º) |    ~14  |            ~200 |
| demais    |     0,1 – 2,3 |              < 6 |

O PCA persegue **variância**, e variância depende da unidade de medida. Sem padronizar, o
primeiro componente principal fica quase colinear ao eixo do `proline` — não porque ele explica
o vinho, mas porque seus números (mg/L, na casa dos milhares) são grandes. Padronizar
(`(X - μ) / σ`) zera essa injustiça: toda feature passa a ter variância 1 e o PCA as compara de
igual pra igual.

## Pipeline

| # | Etapa | Status |
|---|-------|--------|
| 1 | Carregar + padronizar na mão (`normalize`) | ✅ |
| 2 | PCA na mão: covariância → eigendecomposition → projeção (`cov_matrix`, `eig`, `PCA`) | ✅ |
| 3 | Reconstrução (`reconstruct`) — inversa da projeção | ✅ |
| 4 | Validar contra sklearn — testes automatizados em `tests/test_pca.py` (`normalize` centraliza · componentes batem com sklearn · roundtrip do `reconstruct`) | ✅ |
| 5 | Regressão linear do zero: gradiente descendente (`gradient_descent`) — o gradiente do MSE derivado na mão | ✅ |
| 6 | Solução fechada: equação normal (`normal_equation`) + prova de que o GD converge pra ela — testes em `tests/test_regression.py` (GD ≈ eq. normal · gradiente analítico ≈ numérico) | ✅ |
| 7 | Regressão sobre os *scores* do PCA — conecta as duas metades do projeto (provado: MSE idêntico ao da regressão nas features, por invariância a rotação) | ✅ |
| 8 | SVD + comparação com a eigendecomposition (caso mal-condicionado) | ⬜️ |

## Resultados

Variância explicada acumulada (verificada — bate com `sklearn` até a 4ª casa decimal):

| componentes | variância acumulada |
|-------------|--------------------:|
| PC1         |              36,2 % |
| PC1 + PC2   |              55,4 % |
| PC1 + PC2 + PC3 |          66,5 % |
| ~10 de 13   |              ~96 % |

Ou seja: reduzir de 13 → 2 dimensões já preserva mais da metade da variância — o suficiente
pra um scatter 2D colorido por classe.

<!-- TODO: adicionar o gráfico de variância explicada acumulada + o scatter 2D por classe
     quando o notebook de análise estiver pronto. -->

## A matemática por trás

<!-- Renan: rascunho pra você ajustar no seu tom. Confere se sabe defender cada ponto de boca —
     é o que cai em entrevista. -->

**Por que diagonalizar a covariância, e não olhar os dados crus.** A matriz de covariância é
simétrica e positiva semidefinida (PSD). Não é detalhe de livro — são essas duas propriedades que
fazem o PCA existir. Da simetria vêm autovalores reais e autovetores ortonormais (é o que o
`np.linalg.eigh` me entrega). Da PSD vêm autovalores ≥ 0, e isso *tem* que ser verdade: cada
autovalor é uma variância, e variância negativa não existe. Se eu tivesse achado um negativo, era
bug, não descoberta.

**O que autovetor e autovalor significam aqui.** Autovetor é direção; autovalor é quanta variância
mora naquela direção. Os autovetores são os componentes principais, ortogonais entre si — eixos
novos, girados, apontados pra onde os dados mais se espalham. A soma dos autovalores é o traço da
covariância, ou seja, a variância total. O PCA não cria nem apaga variância. Ele só a redistribui,
do eixo que mais explica pro que menos explica.

**Por que `Xnᵀ @ Xn` é a covariância.** Com os dados já centrados, cada entrada `(i,j)` desse
produto soma os produtos dos desvios das features `i` e `j` sobre todas as amostras — que é
literalmente a definição de covariância. Dividir por `N-1` transforma a soma em média.

**Ambiguidade de sinal.** Autovetor é definido a menos de sinal: `v` e `-v` são os dois válidos.
Por isso os meus scores batem com o sklearn em módulo, às vezes com o sinal trocado. Não é bug, é
propriedade — e o teste de validação tem que ser robusto a isso (comparo `np.abs`, não os valores
crus).

**O gradiente do MSE, derivado na mão.** Aqui entra o Cálculo. A loss é `MSE = (1/n)·Σ(ŷ−y)²`, e
pra minimizar preciso da derivada dela em relação aos pesos. Regra da cadeia: a derivada de `r²` é
`2r`, e a derivada do resíduo `r = xᵀw − y` em relação a `w` é o próprio `x`. Junta os dois e o
gradiente vira `(2/n)·Xᵀ(Xw − y)`. Não confiei na fórmula por fé — conferi contra o gradiente
numérico (diferenças finitas) e bateu até a última casa.

**Gradiente descendente vs. equação normal — quando cada um.** Os dois chegam no mesmo `w` (provei
num teste). A diferença é o caminho. A equação normal resolve de uma vez, `w = (XᵀX)⁻¹Xᵀy`, mas
inverte uma matriz: custo ~`O(d³)` e frágil quando as features são quase colineares (o `XᵀX` fica
mal-condicionado). O gradiente descendente chega lá aos poucos, sem inverter nada, e escala pra
tamanhos onde inverter matriz seria inviável. É por isso que o mundo real — e todo deep learning —
roda em gradiente, não em solução fechada. A fórmula fechada aqui serviu de gabarito pra provar
que o meu gradiente estava certo.

**A ponte entre as duas metades.** Regressão linear é invariante a rotação do espaço de features.
PCA com todos os componentes é só uma rotação (base ortonormal, nada perdido — o roundtrip prova).
Então regredir `proline` nos 12 scores dá previsão e MSE idênticos a regredir nas 12 features
originais. Os pesos mudam, porque estão numa base girada; o que o modelo prevê, não. Foi o "aha"
que costurou PCA e regressão num projeto só.

## Como rodar

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

Para rodar os testes:

```bash
python -m pytest -v
```

Os testes validam o PCA contra o `sklearn` (`sklearn.decomposition.PCA`), verificam
propriedades matemáticas próprias (centralização e roundtrip de reconstrução) e cobrem a
regressão (o gradiente descendente converge para a equação normal, o gradiente analítico
bate com o numérico por diferenças finitas, e a regressão sobre os *scores* do PCA reproduz o
MSE da regressão sobre as features — invariância a rotação).

## Estrutura

```
mml-from-scratch/
├── src/         # implementações testáveis (pca.py, regression.py)
├── notebooks/   # a narrativa: EDA → PCA → regressão → comparação
├── tests/       # validação contra sklearn (test_pca.py, test_regression.py)
└── data/        # dataset (Wine vem do próprio sklearn; pasta reservada)
```

## O que aprendi

<!-- Renan: rascunho no seu tom pra ajustar. Tudo aqui saiu de coisa que eu tropecei de verdade. -->

- **Função pura vale mais que script com dados embutidos.** Minha primeira versão do PCA carregava
  o dataset dentro do módulo. Parecia prático — até a hora de testar, quando percebi que não dava
  pra reusar as funções com outra entrada. Tirar o carregamento e deixar os módulos só com funções
  puras foi o que tornou o projeto testável. No `regression.py` já saiu limpo de primeira.

- **Um teste sem `assert` não testa nada.** Óbvio depois, não antes. Meu primeiro teste calculava a
  verificação e jogava o resultado no lixo. O `assert` *é* o teste; o resto é preparação. Aprendi
  também que rodar via `pytest` (que descobre os testes e monta o path) é diferente de rodar o
  arquivo como script — e por que só o primeiro funciona.

- **Num teste de comparação, um lado é o meu código, o outro é a referência.** Perdi um tempo
  comparando o sklearn com ele mesmo transposto antes de sacar que o meu resultado tinha ficado de
  fora do `assert`. Teste que compara a referência com ela mesma é teatro.

- **`.copy()` no NumPy não é opcional.** No gradiente numérico, `w_mais = w` não copia — aponta pro
  mesmo array, e mexer em `w_mais[i]` corrompe o `w` original. Bug silencioso clássico. Hoje
  "referência ou cópia?" é a primeira pergunta que faço quando um array muda sem eu mandar.

- **Derivar o gradiente na mão e provar com diferenças finitas.** Foi o que mais me deu segurança:
  não é "confio que a fórmula tá certa", é "provo que tá". Mesma ideia do backprop, num modelo que
  ainda cabe na cabeça.

- **Detalhes que pareciam pedância viraram entendimento.** O `ddof=1` explica a diferencinha pro
  `StandardScaler`; padronizar (escala) não é o mesmo que centralizar (média); e com os dados
  padronizados o intercepto da regressão dá exatamente a média do alvo — o que usei de sanity check
  pro termo de bias.

- **CI não é enfeite.** Colocar os testes no GitHub Actions me obrigou a garantir que o
  `requirements.txt` está completo, porque o runner parte de um ambiente pelado. Faltou dependência?
  Fica vermelho. (E descobri na marra que push de workflow exige o escopo `workflow` no token.)
