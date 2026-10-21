"""Fine-tuning com o Trainer da Hugging Face."""
import inspect
import json
import math

import matplotlib.pyplot as plt
import torch
from transformers import (EarlyStoppingCallback, Trainer, TrainingArguments,
                          default_data_collator, set_seed)

from src import config


def _argumentos(n_blocos_treino):
    passos_por_epoca = math.ceil(n_blocos_treino / (config.BATCH_SIZE * config.GRAD_ACCUM))
    warmup = int(config.WARMUP_FRAC * passos_por_epoca * config.EPOCAS)
    kw = dict(
        output_dir=str(config.DIR_CHECKPOINTS),
        num_train_epochs=config.EPOCAS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=warmup,
        logging_steps=config.EVAL_STEPS,
        eval_steps=config.EVAL_STEPS,
        save_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        seed=config.SEED,
        report_to="none",
    )
    # O nome do argumento mudou entre versões do transformers.
    nome = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters \
        else "evaluation_strategy"
    kw[nome] = "steps"
    return TrainingArguments(**kw)


def plotar_curvas(log_history, caminho=None):
    treino = [(l["step"], l["loss"]) for l in log_history if "loss" in l]
    val = [(l["step"], l["eval_loss"]) for l in log_history if "eval_loss" in l]
    plt.figure(figsize=(7, 4))
    if treino:
        plt.plot(*zip(*treino), marker="o", label="Loss de treino")
    if val:
        plt.plot(*zip(*val), marker="s", label="Loss de validação")
    plt.xlabel("Passo de treino")
    plt.ylabel("Loss (entropia cruzada)")
    plt.title("Curva de aprendizado do fine-tuning")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    caminho = caminho or config.DIR_GRAFICOS / "curva_loss.png"
    config.DIR_GRAFICOS.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho, dpi=150)
    plt.close()
    return caminho


def treinar(modelo, tokenizer, datasets):
    set_seed(config.SEED)
    trainer = Trainer(
        model=modelo,
        args=_argumentos(len(datasets["train"])),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        # Os datasets já trazem a coluna "labels"; o collator padrão só empilha os tensores.
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.PACIENCIA)],
    )
    trainer.train()

    config.DIR_MODELO.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(config.DIR_MODELO))
    tokenizer.save_pretrained(str(config.DIR_MODELO))
    with open(config.DIR_SAIDA / "log_treino.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    plotar_curvas(trainer.state.log_history)
    print(f"[treino] modelo salvo em {config.DIR_MODELO}")
    return trainer
