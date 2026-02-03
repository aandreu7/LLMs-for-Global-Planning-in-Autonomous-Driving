import json
from transformers import TrainerCallback

class LogLoggerCallback(TrainerCallback):
    """
    Un callback personalizado para guardar los logs (loss, lr, epoch) 
    en un archivo JSONL en tiempo real durante el entrenamiento.
    """
    def __init__(self, log_path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 'logs' es un diccionario con {loss, learning_rate, epoch, etc.}
        if logs:
            # Abrimos en modo 'a' (append) para añadir una línea nueva cada vez
            with open(self.log_path, 'a', encoding='utf-8') as f:
                # Añadimos el step actual al diccionario para tener referencia
                log_entry = {**logs, "step": state.global_step}
                f.write(json.dumps(log_entry) + "\n")