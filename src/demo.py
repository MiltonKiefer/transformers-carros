"""Demonstração interativa: base × ajustado.

Uso:
    python -m src.demo --modo gradio     # interface web (padrão)
    python -m src.demo --modo simples    # loop no terminal
"""
import argparse

from src import config
from src.geracao import gerar
from src.modelo import carregar_modelo, carregar_tokenizer


def carregar_modelos():
    if not config.DIR_MODELO.exists():
        raise FileNotFoundError("Modelo ajustado não encontrado. Rode o treino primeiro.")
    tok = carregar_tokenizer(config.MODELO_BASE)
    base = carregar_modelo(config.MODELO_BASE, tok)
    ajustado = carregar_modelo(str(config.DIR_MODELO), tok)
    return base, ajustado, tok


def demo_simples(base, ajustado, tok):
    """Versão sem Gradio, caso a interface falhe no dia da apresentação."""
    print("Digite um prompt (linha vazia para sair).")
    while True:
        prompt = input("\nPrompt> ").strip()
        if not prompt:
            break
        seed = config.SEED
        print("\n[BASE]    ", gerar(base, tok, prompt, seed=seed)[0])
        print("\n[AJUSTADO]", gerar(ajustado, tok, prompt, seed=seed)[0])


def criar_interface(base, ajustado, tok):
    import gradio as gr

    def responder(prompt, temperatura, top_p, max_tokens, seed):
        kw = dict(temperature=temperatura, top_p=top_p, max_new_tokens=int(max_tokens), seed=int(seed))
        return gerar(base, tok, prompt, **kw)[0], gerar(ajustado, tok, prompt, **kw)[0]

    with gr.Blocks(title="Transformers × Carros") as app:
        gr.Markdown("# 🚗 Transformers × Carros\nGPT-2 original × GPT-2 ajustado com artigos sobre automóveis. "
                    "Mesma seed nos dois modelos, para uma comparação justa.")
        prompt = gr.Textbox(label="Prompt", value="Optimus Prime é um caminhão", lines=2)
        with gr.Row():
            temperatura = gr.Slider(0.1, 1.5, value=config.TEMPERATURA, step=0.05, label="Temperatura")
            top_p = gr.Slider(0.1, 1.0, value=config.TOP_P, step=0.05, label="Top-p")
            max_tokens = gr.Slider(20, 200, value=config.MAX_NEW_TOKENS, step=10, label="Máx. tokens novos")
            seed = gr.Number(value=config.SEED, label="Seed", precision=0)
        botao = gr.Button("Gerar", variant="primary")
        with gr.Row():
            saida_base = gr.Textbox(label="GPT-2 base", lines=8)
            saida_aj = gr.Textbox(label="GPT-2 ajustado (carros)", lines=8)
        botao.click(responder, [prompt, temperatura, top_p, max_tokens, seed], [saida_base, saida_aj])
        gr.Examples([[p] for p in ["Optimus Prime é um caminhão", "O motor de combustão interna funciona",
                                   "A receita de bolo de chocolate começa com",
                                   "Na Fórmula 1, o campeão mundial de 1994 foi"]], inputs=[prompt])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["gradio", "simples"], default="gradio")
    args = ap.parse_args()
    base, ajustado, tok = carregar_modelos()
    if args.modo == "simples":
        demo_simples(base, ajustado, tok)
    else:
        criar_interface(base, ajustado, tok).launch(share=True)


if __name__ == "__main__":
    main()
