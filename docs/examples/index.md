# Como montar o menu

Cada entrega é **um item de menu** em `mkdocs.yml`. O que fica do lado direito dos dois
pontos pode ser um Markdown, um notebook ou uma URL — e o Material renderiza os três do
mesmo jeito na navegação. Itens podem ficar soltos no topo do menu ou agrupados numa
seção (indente-os sob um título, como o `Projeto:` do template).

``` { .yaml .copy title="mkdocs.yml" }
nav:
  - Data: exercises/data/index.md                          # (1)!
  - Perceptron: exercises/perceptron/index.ipynb           # (2)!
  - MLP: https://colab.research.google.com/drive/1AbC      # (3)!
```

1.  **Markdown** — o formato esperado pelo enunciado. Caminho relativo a `docs/`.
2.  **Notebook** — o `.ipynb` é renderizado pelo `mkdocs-jupyter`, com as saídas que
    estiverem salvas no arquivo.
3.  **Link externo** — qualquer URL absoluta vira um item que abre fora do site.

!!! danger "Uma pasta por entrega, e o caminho é fixo"

    Cada entrega mora na própria pasta, com o relatório em `index.md` e o material em
    `code/` e `figures/`: `docs/exercises/<slug>/index.md` e
    `docs/projects/<slug>/index.md`. Os slugs são os do site da disciplina.

    Se você optar por notebook ou Colab, o arquivo entra **na mesma pasta**
    (`docs/exercises/data/index.ipynb`) ou é linkado a partir do `index.md` — não mova a
    entrega para fora dela.

## 1. Relatório em Markdown

O caminho principal. Veja [1. Data](../exercises/data/index.md) preenchido como modelo:
*front matter*, títulos espelhando o enunciado, figura legendada, código incluído a partir
de `code/` e tabela *Results summary*.

Para trazer um script para dentro do relatório sem copiar e colar:

```` { .markdown .copy title="docs/exercises/data/index.md" }
``` { .python .copy linenums='1' title="exercise1.py" }
;--8<-- "docs/exercises/data/code/exercise1_point_clouds.py"
```
````

O caminho é relativo à **raiz do repositório** (`base_path: [.]`). Se o arquivo não
existir, o build falha — o que é bom: você descobre o link quebrado antes do professor.

## 2. Notebook direto no menu

Coloque o `.ipynb` dentro de `docs/` e aponte a nav para ele:

``` { .yaml .copy title="mkdocs.yml" }
nav:
  - Notebook no menu: examples/notebook/data-exercise-1.ipynb
```

O resultado está em [Notebook no menu](notebook/data-exercise-1.ipynb).

!!! warning "Salve o notebook com as saídas"

    O build roda com `execute: false` — nada é executado na publicação. Gráficos e
    resultados só aparecem no site se estiverem **salvos dentro do `.ipynb`** commitado.
    Rode todas as células, salve, e só então faça o commit.

## 3. Link para o Colab

Uma URL absoluta na nav vira um item que abre em outra aba:

``` { .yaml .copy title="mkdocs.yml" }
nav:
  - Abrir no Colab ↗: https://colab.research.google.com/github/usuario/ann-dl/blob/main/docs/examples/notebook/data-exercise-1.ipynb
```

Duas formas de montar o endereço:

=== "Notebook do repositório"

    O Colab abre qualquer notebook público do GitHub trocando o domínio:

    ```
    https://colab.research.google.com/github/<usuario>/<repo>/blob/<branch>/<caminho>.ipynb
    ```

    Vantagem: o notebook continua versionado no seu repositório — o professor vê o mesmo
    arquivo no site, no Git e no Colab.

=== "Notebook do Google Drive"

    Use **Compartilhar → Qualquer pessoa com o link** e copie o endereço:

    ```
    https://colab.research.google.com/drive/<ID_DO_NOTEBOOK>
    ```

    !!! failure "Cuidado"

        Um notebook que mora só no Drive **não** faz parte do repositório. Como o prazo é
        o timestamp do último commit e a correção lê o repositório, exporte o `.ipynb`
        para `docs/` de qualquer maneira.

Para colocar o badge dentro de uma página, em vez de no menu:

``` { .markdown .copy }
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/usuario/ann-dl/blob/main/docs/examples/notebook/data-exercise-1.ipynb){:target='_blank'}
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/usuario/ann-dl/blob/main/docs/examples/notebook/data-exercise-1.ipynb){:target='_blank'}

## Recursos disponíveis no template

=== "Figuras"

    ``` { .markdown .copy }
    ![Texto alternativo](figures/fig01-exemplo.svg)
    /// caption
    **Figura 1** — legenda descrevendo o que a figura mostra.
    ///
    ```

    ![Exemplo de figura](../exercises/data/figures/fig01-exemplo.svg)
    /// caption
    **Figura 1** — clique na imagem para ampliar (`glightbox`).
    ///

=== "Matemática"

    ``` { .markdown .copy }
    A função de perda é $\mathcal{L} = -\sum_i y_i \log \hat{y}_i$, e a atualização:

    $$
    w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}
    $$
    ```

    A função de perda é $\mathcal{L} = -\sum_i y_i \log \hat{y}_i$, e a atualização:

    $$
    w \leftarrow w - \eta \frac{\partial \mathcal{L}}{\partial w}
    $$

=== "Diagramas"

    ```` { .markdown .copy }
    ``` mermaid
    flowchart LR
        x[Entrada] --> h[Oculta] --> y[Saída]
    ```
    ````

    ``` mermaid
    flowchart LR
        x[Entrada] --> h[Oculta] --> y[Saída]
    ```

    Monte os seus no [Mermaid Live Editor](https://mermaid.live/){:target='_blank'}.

=== "Checklists"

    ``` { .markdown .copy }
    - [x] Exercise 1
    - [ ] Exercise 2
    ```

    - [x] Exercise 1
    - [ ] Exercise 2
