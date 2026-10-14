# Redes Neurais Artificiais & Deep Learning

???+ info inline end "Edição"

    **20XX.X** <!-- TROCAR -->

    [Enunciados :material-open-in-new:](https://insper.github.io/ann-dl/){:target='_blank'}

Este site é o **portfólio** das entregas da disciplina. Ele cresce ao longo do semestre:
cada exercício e cada projeto vira um item de menu, e o repositório que o gera é parte
da avaliação — o professor lê o site publicado **e** o repositório (Markdown, código e
histórico do Git).

## Identificação

Quem responde por este repositório. Os **exercícios são individuais**; a equipe do projeto
— que pode ser diferente — fica registrada na [página do projeto](projects/index.md).

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| | | |

!!! tip "Como usar este template"

    Este é um **bloco de notas versionado**: registre o que foi feito, o que falta e as
    decisões tomadas, commitando a cada avanço. O prazo de uma entrega é o *timestamp do
    último commit que toca a pasta daquela entrega* — não a hora do formulário nem a da
    publicação no Pages.

    Comece por [Como usar este template](template/index.md).

## Status das entregas

!!! info "Datas, pesos e regras são da sua edição"

    O [overview da edição](https://insper.github.io/ann-dl/){:target='_blank'} traz o
    calendário, os pesos de cada entrega e as regras de avaliação. Este template não os
    repete — copie para cá o que a sua turma precisa acompanhar, ou mantenha só o status.

    A lista abaixo é o conjunto usual de entregas; acrescente ou remova itens conforme a
    sua edição, ajustando também as pastas em `docs/` e a `nav` do `mkdocs.yml`.

### Exercícios — individuais

- [ ] [Data](exercises/data/index.md)
- [ ] [Perceptron](exercises/perceptron/index.md)
- [ ] [MLP](exercises/mlp/index.md)
- [ ] [VAE](exercises/vae/index.md)

### [Projeto](projects/index.md) — em equipe

Um projeto, um dataset, três entregas:

- [ ] [EDA](projects/eda/index.md)
- [ ] [Classificação](projects/classification/index.md) **ou** [Regressão](projects/regression/index.md)
- [ ] [Generativo](projects/generative/index.md)

## Checklist antes de cada entrega

- [ ] Repositório **público** e o GitHub Pages construindo sem erro.
- [ ] Caminho correto: `docs/exercises/<slug>/index.md` (ou `docs/projects/<slug>/index.md`).
- [ ] *Front matter* com `exercise:` (ou `project:`) e `ai_use:` preenchidos.
- [ ] Títulos espelhando a estrutura do enunciado (`## Exercise N`, `### A`, `### B`, ...).
- [ ] Figuras commitadas em `figures/`, numeradas e exibidas no relatório.
- [ ] Scripts como arquivos reais em `code/`, referenciados via `--8<--`.
- [ ] Tabela **Results summary** completa, sem linhas em branco.
- [ ] Último commit anterior ao prazo.

!!! danger "Escreva para defender"

    As notas da disciplina costumam estar sujeitas a defesa oral, e a nota do projeto, a uma
    prova sobre o próprio projeto. Escreva relatórios que você consiga sustentar meses
    depois — o que inclui entender cada linha do código que está no repositório. Confira as
    regras da sua edição no overview.

!!! danger "Uso de IA"

    O campo `ai_use` é **obrigatório** em toda entrega. Colaborar com IA é permitido;
    não declarar o uso, não. Descreva o que foi gerado, revisado ou depurado com apoio de
    IA — ou escreva `"none"`.
