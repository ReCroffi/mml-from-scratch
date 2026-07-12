# MML from scratch

**🌐 Idioma:** **Português** · [English](README_en.md)

> Implementação do zero (NumPy) de PCA e regressão linear, aplicada a um dataset real —
> encerrando a especialização *Mathematics for Machine Learning* (Imperial College / Coursera).

> ⚠️ **Em construção.** O PCA está implementado e validado contra o sklearn. Regressão,
> testes formais e comparação com SVD estão nas próximas etapas (ver [Pipeline](#pipeline)).

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
| 4 | Validar contra sklearn (variância explicada, scores, reconstrução) | 🔄 informal feito, falta formalizar em `tests/` |
| 5 | Regressão via gradiente descendente nos dados reduzidos | ⬜️ |
| 6 | Comparar com solução fechada (equação normal) | ⬜️ |
| 7 | SVD + comparação com a eigendecomposition (caso mal-condicionado) | ⬜️ |

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

<!-- Renan: este conteúdo foi montado a partir do que VOCÊ argumentou durante o desenvolvimento.
     Reescreva no seu tom e confira se sabe defender cada ponto de boca — é o que cai em
     entrevista. Não decore a minha frase. -->

- **Por que a matriz de covariância, e não os dados crus?** Porque ela é **simétrica** e
  **positiva semidefinida (PSD)**. Da simetria vêm autovalores **reais** e autovetores
  **ortonormais** (via `np.linalg.eigh`); da PSD vêm autovalores **≥ 0** — o que faz sentido
  físico, já que cada autovalor é uma variância, e variância não pode ser negativa.

- **O que são autovetores e autovalores aqui?** Os **autovetores** são as direções de máxima
  variância (os componentes principais, ortogonais entre si); os **autovalores** dizem *quanta*
  variância há em cada direção. A soma dos autovalores é igual ao traço da covariância — a
  variância total é **conservada**, o PCA apenas a redistribui sobre eixos ortogonais.

- **Por que `Xnᵀ @ Xn` é a covariância?** Com os dados já centrados, cada entrada `(i,j)` desse
  produto soma os produtos dos desvios das features `i` e `j` sobre todas as amostras — a própria
  definição de covariância; dividir por `N-1` transforma soma em média.

- **Ambiguidade de sinal.** Autovetores são definidos a menos de sinal (`v` e `-v` são ambos
  válidos). Por isso os scores batem com o sklearn *em módulo*, com o sinal eventualmente trocado
  — não é bug, é propriedade. Testes de validação precisam ser robustos a isso.

<!-- TODO (após implementar): gradiente descendente vs. equação normal — quando cada um. -->

## Como rodar

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
```

Para rodar os testes (quando existirem):

```bash
pytest
```

## Estrutura

```
mml-from-scratch/
├── src/         # implementações testáveis (pca.py; regression.py em breve)
├── notebooks/   # a narrativa: EDA → PCA → regressão → comparação
├── tests/       # validação contra sklearn
└── data/        # dataset (Wine vem do próprio sklearn; pasta reservada)
```

## O que aprendi

<!-- Renan: 3-4 bullets HONESTOS, no seu tom. Puxa do que você tropeçou de verdade nessas
     sessões. Sugestões de tópicos que renderam bons "aha" (escreva você):
     - a diferença entre padronizar (escala) e centralizar, e por que o Wine exige as duas
     - por que `eigh` garante autovalores REAIS, mas a não-negatividade vem da covariância ser PSD
       (são fontes diferentes — não confundir)
     - a escolha de ddof (1 vs 0) e como ela explica a diferencinha ao comparar com o StandardScaler
     - por que SVD é mais estável que a eigendecomposition da covariância (elevar ao quadrado
       amplifica erro de arredondamento) — a preencher quando implementar o SVD -->
