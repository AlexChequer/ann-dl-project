# Projeto

!!! abstract "Enunciados"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

O projeto é **um só**, feito em equipe sobre **o mesmo dataset**, e entregue em três partes
ao longo do semestre, cada uma com data e peso próprios.

## Equipe

!!! danger "Preencha antes de qualquer entrega"

    Toda entrega do projeto é avaliada em equipe. Se os nomes não estiverem aqui, não há
    como atribuir a nota — e o mesmo vale para o `mkdocs.yml`, cujo `site_author` deve
    listar o grupo.

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| Alex Chequer | | [AlexChequer](https://github.com/AlexChequer) |
| | | [Lucasikawa](https://github.com/Lucasikawa) |
| | | [carloshernanic](https://github.com/carloshernanic) |

Times de 2 a 3 pessoas. Repita esses nomes no cabeçalho de cada entrega — quem corrige pode
abrir uma página sozinha, sem passar por aqui.

## As três entregas

| # | Entrega | Página |
|---|---------|--------|
| 1 | EDA | [EDA](eda/index.md) |
| 2 | Classificação **ou** Regressão | [Regressão](regression/index.md) |
| 3 | Generativo | [Generativo](generative/index.md) |

Datas e pesos são da sua edição — veja o
[overview](https://insper.github.io/ann-dl/){:target='_blank'}.

!!! danger "A nota do projeto costuma ser limitada por uma prova sobre o próprio projeto"

    Deliverables bem escritos não sustentam uma equipe que não consegue explicar o que
    entregou. Escreva os relatórios de modo que você consiga defendê-los meses depois, e
    confira no overview da sua edição como a prova entra na nota.

!!! warning "Escolha uma: classificação ou regressão"

    A segunda entrega é **uma das duas**, não as duas. Este template traz as duas pastas
    para você escolher; depois de decidir, apague a que não vai usar — da pasta `docs/projects/`
    **e** da `nav` no `mkdocs.yml`.

## Dataset

O mesmo dataset atravessa as três entregas — escolhê-lo bem no EDA é o que torna as outras
duas viáveis.

| | |
|---|---|
| **Nome** | New York City Airbnb Open Data (`AB_NYC_2019.csv`) |
| **Fonte (URL)** | <https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data> (dados do Inside Airbnb) |
| **Licença / termos de uso** | CC0 1.0 (domínio público) |
| **Amostras** | 48.895 anúncios (48.884 após remover `price == 0`) |
| **Features** | 16 colunas no CSV; 7 numéricas + 3 categóricas usadas no modelo |
| **Variável alvo** | `price` — diária em US$ (modelada como `log1p(price)`) |
| **Tarefa escolhida** | Regressão |

Prever a diária de um anúncio a partir de onde ele está, que tipo de espaço oferece e do seu
histórico de reviews/disponibilidade. Não é trivial: o alvo tem cauda muito longa (até
US$ 10.000), a localização tem 220 bairros com muitos raros, e nenhuma feature numérica
sozinha explica bem o preço (maior |ρ| = 0,44). O CSV está versionado em `data/`.

## Status

- [x] **1. EDA**
- [ ] **2. Regressão**
- [ ] **3. Generativo**

## Registro de decisões

Anote aqui as decisões que atravessam as entregas — troca de dataset, mudança de alvo,
recorte de features — com a data. É o que permite reconstruir o raciocínio na prova de
projeto.

| Data | Decisão | Motivo |
|------|---------|--------|
| 2026-10-14 | Dataset NYC Airbnb 2019, regressão de `price` | ver EDA |
| 2026-10-14 | Alvo modelado como `log1p(price)` | assimetria 19,22 → 0,64 |
| 2026-10-14 | Pipeline em `docs/projects/eda/code/pipeline.py` reutilizado nas próximas entregas | fit só no treino, 102 features |
