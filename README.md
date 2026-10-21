# 🚗 Transformers × Carros

Fine-tuning de um **GPT-2 pequeno em português** (`pierreguillou/gpt2-small-portuguese`, ~124M de parâmetros)
com artigos da Wikipédia sobre automóveis. O tema é um trocadilho com o assunto da apresentação
(*Transformers*): o projeto mostra, de forma prática, o paradigma **pré-treinar e depois adaptar**.

## Sumário
1. [Objetivo e conceitos demonstrados](#objetivo-e-conceitos-demonstrados)
2. [Estrutura do projeto](#estrutura-do-projeto)
3. [Como executar](#como-executar)
4. [Arquitetura](#arquitetura)
5. [Componentes](#componentes)
6. [Como ler as métricas](#como-ler-as-métricas)
7. [Usando outra base de dados](#usando-outra-base-de-dados)
8. [Limitações conhecidas](#limitações-conhecidas)
9. [Referências](#referências)

## Objetivo e conceitos demonstrados
| Conceito | Onde aparece |
|---|---|
| Transformer *decoder-only* (GPT-2) | `src/modelo.py` (`resumo_modelo` mostra camadas, cabeças, d_model) |
| Treino autorregressivo (prever o próximo token) | `src/preprocessamento.py` (`labels = input_ids`) e `src/treino.py` |
| Transfer learning / fine-tuning | comparação base × ajustado |
| Perplexidade e overfitting | `src/avaliacao.py` e curva de loss |
| Esquecimento (*catastrophic forgetting*) | perplexidade fora do domínio |
| Amostragem (temperatura, top-k, top-p) | `src/geracao.py` e demo |
| Alucinação e dependência dos dados | prompts "armadilha" (Ferrari 250 GTO, F1 1994, Optimus Prime) |

## Estrutura do projeto
```
transformers-carros/
├── README.md
├── requirements.txt
├── notebook_colab.ipynb        # roteiro completo, ideal para a apresentação
├── run_pipeline.py             # executa o pipeline fora do notebook
├── src/
│   ├── config.py               # hiperparâmetros, caminhos, listas de temas
│   ├── coleta_dados.py         # coleta na API da Wikipédia
│   ├── fontes_dados.py         # carrega dados de várias fontes (wikipedia, txt, csv, HF)
│   ├── preprocessamento.py     # limpeza, split por artigo, tokenização em blocos
│   ├── modelo.py               # tokenizador e GPT-2
│   ├── treino.py               # Trainer, early stopping, curva de loss
│   ├── avaliacao.py            # perplexidade, vocabulário, diversidade, gráficos
│   ├── geracao.py              # gerar() e comparar(); prompts de teste
│   └── demo.py                 # interface Gradio e versão simples
├── data/                       # corpus bruto e split (criados na execução)
└── outputs/
    ├── modelo_carros/          # modelo ajustado
    ├── metricas.json
    ├── log_treino.json
    └── graficos/               # curva_loss.png, perplexidade.png
```

## Como executar

### Opção A: Google Colab (recomendada)
1. Compacte a pasta em `transformers-carros.zip`.
2. Abra `notebook_colab.ipynb` no [Google Colab](https://colab.research.google.com) (*Arquivo → Fazer upload de notebook*).
3. Ative a GPU: *Ambiente de execução → Alterar tipo de ambiente → GPU T4*.
4. Execute as células em ordem. Na célula "Enviar o projeto", selecione o `.zip`.
5. A coleta leva alguns minutos, o treino 5 a 15 min na T4.

### Opção B: máquina local
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py --etapa tudo                    # coleta + treino + avaliação + comparação
python run_pipeline.py --etapa demo                    # abre a interface Gradio
# ou etapas isoladas:
python run_pipeline.py --etapa coleta | treino | avaliacao | comparar
python -m src.demo --modo simples                      # demo no terminal, sem Gradio
```
Sem GPU o treino funciona, mas é bem mais lento. Se faltar memória, reduza `BATCH_SIZE` em `src/config.py`
(e aumente `GRAD_ACCUM` para manter o tamanho efetivo do lote).

### Dicas para a apresentação
- Treine **antes**. O modelo fica salvo em `outputs/modelo_carros/`; na demo basta carregá-lo.
- O corpus fica salvo em `data/`, então a API da Wikipédia não é necessária no dia.
- Se o modelo em português não carregar, troque `MODELO_BASE` por `MODELO_ALTERNATIVO` (`distilgpt2`) em `config.py`
  (os prompts e textos de avaliação ficam em português, então use isso só como plano B).

## Arquitetura

```
Wikipédia (API) ──► coleta_dados ──► artigos_carros.jsonl
                                         │
                                         ▼
                    preprocessamento: limpeza + split por artigo (90% / 10%)
                                         │
                                         ▼
          tokenizador GPT-2 ──► concatena (EOS entre artigos) ──► blocos de 256 tokens
                                         │
           ┌─────────────────────────────┴──────────────────────────┐
           ▼                                                        ▼
  GPT-2 base (referência, sem ajuste)               GPT-2 + fine-tuning (Trainer)
           │                                                        │
           └───────────────► avaliacao ◄─────────────────────────────┘
                 perplexidade · vocabulário · diversidade · repetição
                                         │
                                         ▼
                        geracao / demo: prompt → base × ajustado
```

**Dentro do modelo** (o mesmo diagrama de Transformer da apresentação): tokens → embeddings de token + posição →
N blocos *decoder* (auto-atenção multi-cabeça com máscara causal → LayerNorm → MLP → conexões residuais) →
LayerNorm → camada linear → *softmax* sobre o vocabulário. O fine-tuning ajusta **todos** os pesos
com a mesma tarefa do pré-treino (prever o próximo token), só que com dados do domínio.

## Componentes

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | Fonte única de verdade: modelo, caminhos, lista de títulos/categorias/buscas, hiperparâmetros de treino e geração. |
| `coleta_dados.py` | Reúne títulos candidatos (semente + categorias + buscas), baixa o texto puro de cada artigo, remove seções finais (Referências, Ligações externas...), descarta artigos curtos e salva em JSONL. |
| `fontes_dados.py` | Camada de abstração: qualquer fonte vira `[{"titulo", "texto"}]`. Fontes: `wikipedia`, `pasta_txt`, `csv`, `huggingface`. |
| `preprocessamento.py` | Limpeza de espaços, divisão treino/validação **por documento** (evita vazamento), tokenização, concatenação com EOS e fatiamento em blocos fixos. Produz um `DatasetDict` com `input_ids` e `labels`. |
| `modelo.py` | Carrega tokenizador e `AutoModelForCausalLM`, define `pad_token` (GPT-2 não tem), move para GPU e imprime o resumo da arquitetura. |
| `treino.py` | `Trainer` com AdamW, warm-up, fp16, avaliação a cada 25 passos, *early stopping* e carregamento do melhor checkpoint. Salva modelo, log e curva de loss. |
| `avaliacao.py` | Perplexidade (validação e fora do domínio), taxa de vocabulário automotivo, distinct-n e repetição. Gera `metricas.json` e gráficos. |
| `geracao.py` | `gerar()` com temperatura/top-k/top-p/seed e `comparar()` lado a lado. Define os prompts de domínio e de "armadilha". |
| `demo.py` | Interface Gradio (dois painéis, mesma seed) e versão simples no terminal. |

## Como ler as métricas
- **Perplexidade (validação)**: quão "surpreso" o modelo fica com textos de carros que não viu. Esperado: **cai** após o fine-tuning.
- **Perplexidade (fora do domínio)**: mesma medida em textos de culinária, futebol e biologia. Esperado: **sobe** um pouco, o que indica esquecimento/especialização.
- **Vocabulário automotivo**: fração de palavras geradas ligadas a carros (lista de termos e raízes em `avaliacao.py`; é uma medida aproximada).
- **Distinct-1/2**: diversidade lexical (maior = mais variado).
- **Repetição (4-gramas)**: quanto o texto se repete (maior = pior).
- **Curva de loss**: se a loss de validação voltar a subir enquanto a de treino cai, há *overfitting*.

Os números exatos variam conforme o corpus coletado e a seed. Use os valores do **seu** `metricas.json` na apresentação.

## Usando outra base de dados
Edite `FONTE_DADOS` em `src/config.py`:

| Valor | O que fazer |
|---|---|
| `"pasta_txt"` | Coloque arquivos `.txt` (UTF-8) em `data/custom_txt/`. Cada arquivo é um documento. Um arquivo único grande é dividido automaticamente. |
| `"csv"` | Crie `data/custom.csv` com uma coluna de texto e ajuste `COLUNA_TEXTO_CSV`. |
| `"huggingface"` | Defina `HF_DATASET`, `HF_COLUNA_TEXTO` e, se quiser filtrar o tema, `HF_PALAVRAS_CHAVE`. |

Depois, apague `outputs/modelo_carros/` se quiser retreinar do zero e execute o pipeline novamente.
Verifique sempre a **licença** dos dados. Os artigos da Wikipédia são CC BY-SA.

## Limitações conhecidas
- Modelo pequeno (124M): coerência curta, repetições e frases que "parecem boas" mas não fazem sentido.
- **Alucinação**: o modelo gera fatos plausíveis e falsos (anos, campeões, especificações). Ele aprende estilo e estatística, não verdade.
- Corpus pequeno (~1 a 2 MB): risco de memorização e de viés dos artigos escolhidos.
- O fine-tuning pode degradar o desempenho fora do domínio (esquecimento).
- A taxa de vocabulário automotivo usa uma lista fixa de termos, então é só um indicador.
- O resultado da coleta depende da Wikipédia no momento da execução (artigos mudam).

## Referências
- Vaswani et al. (2017). *Attention Is All You Need*.
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2).
- Documentação do Hugging Face: [Transformers](https://huggingface.co/docs/transformers), [Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling), [Datasets](https://huggingface.co/docs/datasets).
- Modelo base: [`pierreguillou/gpt2-small-portuguese`](https://huggingface.co/pierreguillou/gpt2-small-portuguese).
- Dados: [Wikipédia em português](https://pt.wikipedia.org), licença CC BY-SA 4.0.
