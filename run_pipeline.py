"""Executa o pipeline completo (ou etapas isoladas) fora do notebook.

    python run_pipeline.py --etapa tudo
    python run_pipeline.py --etapa coleta | treino | avaliacao | comparar | demo
"""
import argparse

from src import config
from src.fontes_dados import carregar_documentos
from src.modelo import carregar_modelo, carregar_tokenizer, resumo_modelo
from src.preprocessamento import preparar_datasets


def _datasets(tok):
    return preparar_datasets(carregar_documentos(), tok)


def etapa_coleta():
    carregar_documentos()


def etapa_treino():
    from src.treino import treinar
    tok = carregar_tokenizer()
    modelo = carregar_modelo(config.MODELO_BASE, tok)
    resumo_modelo(modelo)
    treinar(modelo, tok, _datasets(tok))


def etapa_avaliacao():
    from src.avaliacao import avaliar_tudo
    tok = carregar_tokenizer()
    base = carregar_modelo(config.MODELO_BASE, tok)
    ajustado = carregar_modelo(str(config.DIR_MODELO), tok)
    avaliar_tudo(base, ajustado, tok, _datasets(tok))


def etapa_comparar():
    from src.geracao import comparar
    tok = carregar_tokenizer()
    base = carregar_modelo(config.MODELO_BASE, tok)
    ajustado = carregar_modelo(str(config.DIR_MODELO), tok)
    comparar(base, ajustado, tok)


def etapa_demo():
    from src.demo import main
    main()


ETAPAS = {"coleta": etapa_coleta, "treino": etapa_treino, "avaliacao": etapa_avaliacao,
          "comparar": etapa_comparar, "demo": etapa_demo}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", default="tudo", choices=["tudo", *ETAPAS])
    args = ap.parse_args()
    if args.etapa == "tudo":
        for nome in ("coleta", "treino", "avaliacao", "comparar"):
            print(f"\n######## {nome.upper()} ########")
            ETAPAS[nome]()
        print("\nPipeline concluído. Para a demonstração: python run_pipeline.py --etapa demo")
    else:
        ETAPAS[args.etapa]()
