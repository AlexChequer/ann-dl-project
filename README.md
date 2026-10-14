# Template de entregas — Redes Neurais Artificiais & Deep Learning

Template de site (MkDocs Material + GitHub Pages) para as entregas da disciplina —
[enunciados](https://insper.github.io/ann-dl/). Serve a **qualquer edição**: datas, pesos e a
lista de entregas ficam no site da disciplina, não aqui.

Cada entrega é um item de menu, e o alvo do item pode ser um relatório em Markdown, um
notebook `.ipynb` ou um link do Google Colab. Os três casos estão demonstrados na seção
**Exemplos de uso** do site.

## Estrutura

```
docs/
  index.md                     # capa: grupo e status das entregas
  template/index.md            # como usar este template
  exercises/
    data/{index.md,code/,figures/}
    perceptron/{index.md,code/,figures/}
    mlp/{index.md,code/,figures/}
    vae/{index.md,code/,figures/}
  projects/
    index.md                   # equipe, dataset, decisões, status
    eda/{index.md,code/,figures/}
    classification/{index.md,code/,figures/}
    regression/{index.md,code/,figures/}
    generative/{index.md,code/,figures/}
  examples/                    # exemplos de uso do menu (pode ser removido)
```

Os slugs de `exercises/` e `projects/` são fixos e casam com o site da disciplina. Não os
renomeie.

Cada entrega tem a própria pasta, sempre no mesmo formato: `index.md` para o relatório,
`code/` para os scripts e `figures/` para as imagens. Um notebook entra na mesma pasta,
como `index.ipynb`.

Os nomes do grupo vão em [docs/projects/index.md](docs/projects/index.md) e se repetem no
cabeçalho de cada entrega do projeto.

## As entregas

O conjunto abaixo é o das edições recentes; confira o overview da **sua** edição e ajuste as
pastas em `docs/` junto com a `nav` do `mkdocs.yml`.

- **Exercícios**, individuais: Data, Perceptron, MLP, VAE.
- **Projeto**, em equipe: um único dataset em três entregas — EDA, Classificação **ou**
  Regressão, e Generativo.

O template traz as pastas de classificação e regressão; apague a que a equipe não escolher,
da pasta e da `nav`.

## Setup

```shell
python3 -m venv env
source ./env/bin/activate          # Windows: .\env\Scripts\activate
python3 -m pip install -r requirements.txt --upgrade
```

## Rodando localmente

```shell
mkdocs serve -o
```

## Publicação

O workflow em [.github/workflows/main.yaml](.github/workflows/main.yaml) roda
`mkdocs gh-deploy --force` a cada push na `main`: ele constrói o HTML, empurra para a branch
`gh-pages`, e é essa branch que o GitHub Pages serve.

Configuração inicial, uma vez:

1. **Se você forkou**, habilite os workflows na aba **Actions** (forks vêm com o Actions
   desligado). Usando *Use this template* isso não é necessário.
2. Troque no [mkdocs.yml](mkdocs.yml) todas as linhas marcadas com `# TROCAR`
   (`grep -n TROCAR mkdocs.yml`).
3. Nada a fazer quanto a permissões: o workflow já declara `permissions: contents: write`.
   Só se o build falhar com `Permission denied to github-actions[bot]` vá em
   **Settings → Actions → General → Workflow permissions** → **Read and write permissions**.
4. Dê o primeiro push e espere o run terminar — é ele que cria a branch `gh-pages`.
5. **Settings → Pages** → *Deploy from a branch* → branch **`gh-pages`**, pasta **`/ (root)`**.
   Apontar o Pages para a `main` publica o Markdown cru, não o site.

O passo a passo com as telas está em
[Como usar este template → Publicação no GitHub Pages](docs/template/index.md).

Antes de dar push, valide localmente — o CI publica mesmo com avisos, o modo estrito não:

```shell
mkdocs build --strict
```

Para publicar manualmente, sem passar pelo CI:

```shell
mkdocs gh-deploy
```

## Prazo

O prazo de uma entrega é o **timestamp do último commit que toca a pasta daquela entrega**
— não a hora do formulário nem a da publicação. Commite progressivamente.
