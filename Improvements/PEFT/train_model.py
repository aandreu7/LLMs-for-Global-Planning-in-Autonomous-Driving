import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from trainer_callback import LogLoggerCallback

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
from collections import Counter
import argparse
import re


"""
NOTA IMPORTANTE:

SFTTrainer --> YA NO LO USAMOS. No soporta enmascaramiento (labels) automático para LMMs.

Trainer --> USAMOS ESTE, más complejo pero más personalizable. Creamos nosotros el DataCollator para el enmascaramiento.
"""


# --- 1. LA LÓGICA DE ENMASCARAMIENTO ---

def find_assistant_start(base_model_name, input_ids, processor):
    """
    Busca dónde empieza la respuesta del asistente para enmascarar lo anterior.
    """
    # Estos IDs dependen del tokenizer de Qwen. 
    # Generalmente es la secuencia: [im_start_id, assistant_id]
    # Lo buscamos dinámicamente para no hardcodear números.

    if base_model_name in ["Qwen/Qwen3-VL-32B-Instruct", "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"]:
        im_start_id = processor.tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        assistant_id = processor.tokenizer.encode("assistant", add_special_tokens=False)[0]
    elif base_model_name == "google/gemma-3-12b-it":
        im_start_id = processor.tokenizer.encode("<start_of_turn>", add_special_tokens=False)[0]
        assistant_id = processor.tokenizer.encode("model", add_special_tokens=False)[0]
    else:
        raise ValueError(f"Modelo no soportado para find_assistant_start: {base_model_name}")
    
    # Buscamos la secuencia [im_start, assistant]
    for i in range(len(input_ids) - 1):
        if input_ids[i] == im_start_id and input_ids[i+1] == assistant_id:
            # Encontramos el inicio. 
            # Sumamos un poco más para saltar el newline (\n) si existe, 
            # pero con marcar desde aquí + 2 o +3 suele bastar.
            # Vamos a retornar el índice justo donde empieza el CONTENIDO de la respuesta.
            return i + 3 # +2 para saltar '<|im_start|', 'assistant' y '\n'
            
    return 0 # Si no lo encuentra (no debería pasar), no enmascara nada (fallback)

class DataCollatorForVisualSFT:
    def __init__(self, model_id, processor):
        self.model_id = model_id
        self.processor = processor
        self.tokenizer = processor.tokenizer
        
        # 1. CALCULAR IDs DINÁMICAMENTE (Seguridad)
        # Probamos con y sin espacio por si acaso, aunque tras \n suele ser sin espacio.
        candidates = ["Straight", "Right", "Left"]
        self.valid_ids = set()
        otros_invalidos = set()
        for w in candidates:
            # Codificar palabra limpia
            self.valid_ids.add(self.tokenizer.encode(w, add_special_tokens=False)[0])
            # Codificar con espacio (por si el prompt dejase un espacio al final)
            otros_invalidos.add(self.tokenizer.encode(" " + w, add_special_tokens=False)[0])
        
        print(f"✅ DataCollator inicializado. IDs válidos para respuesta: {self.valid_ids}")
        print(f"IDs inválidos que pueden ser confundidos: {otros_invalidos}")
        
        # Variable para debug (solo imprimir la primera vez)
        self.debug_printed = False

    def __call__(self, examples):
        texts = []
        images = []
        
        for example in examples:
            text = self.processor.apply_chat_template(
                example["messages"], 
                tokenize=False,
                add_special_tokens=False, # Gemma adds additional <bos> tokens if set to True, therefore, always set to False. No effect for Qwen.
                add_generation_prompt=False # Always set to False, as the conversational format already includes the generation prompt
            )
            texts.append(text)
            images.append(example["images"])

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            add_special_tokens=False, # Gemma adds additional <bos> tokens if set to True, therefore, always set to False. No effect for Qwen.
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        
        # Máscara de padding estándar
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        for i in range(len(labels)): # Con batch_size = 1, solo itera una única vez
            # Encuentra el índice del PRIMER token de la respuesta
            start_idx = find_assistant_start(self.model_id, batch["input_ids"][i], self.processor)
            
            # Verificación de seguridad
            token_id = batch["input_ids"][i, start_idx].item()
            if token_id not in self.valid_ids:
                decoded = self.tokenizer.decode([token_id])
                raise ValueError(f"Sample {i}: Se esperaba Straight/Right/Left pero se encontró ID {token_id} ('{decoded}')")

            # 1. Enmascarar todo antes de la respuesta
            labels[i, :start_idx] = -100

            # 2. Enmascarar lo que viene después (Limpieza agresiva)
            # Asumimos [Palabra] + [im_end] + [\n] = 3 tokens.
            # Enmascaramos desde el 4º token en adelante.
            labels[i, start_idx+3:] = -100
        
        batch["labels"] = labels

        # --- DEBUGGING SOLO LA PRIMERA VEZ ---
        if not self.debug_printed:
            print("\n🔍 --- INSPECCIÓN DATA COLLATOR (Primer Batch) ---")
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"Labels shape: {batch['labels'].shape}")
            print(f"Pixel values shape: {batch['pixel_values'].shape}")
            
            # Imprimir el primer ejemplo decodificado mostrando qué se entrena
            ex_idx = 0
            ids = batch['input_ids'][ex_idx]
            labs = batch['labels'][ex_idx]
            imgs_vals = batch['pixel_values'][ex_idx]

            print("\ninput_ids: ", ids, "\n")
            print("\nlabels: ", labs, "\n")
            print("\npixel_values: ", imgs_vals, "\n")
            
            print("\nVerificando tokens entrenables (Label != -100):")
            for idx, (token, label) in enumerate(zip(ids, labs)):
                if label != -100:
                    print(f"idx {idx}: '{self.tokenizer.decode([token])}' (ID: {token}) <-- APRENDIENDO")
            print("-------------------------------------------------\n")
            self.debug_printed = True

        return batch


def verify_trainable_layers(model):
    """
    Imprime un informe detallado sobre qué partes del modelo (Visión vs Lenguaje)
    se están entrenando. Versión mejorada para Gemma-3 y Qwen-3-VL.
    """
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    
    if not trainable_params:
        print("WARNING: No se encontraron parametros entrenables. El modelo esta congelado.")
        return

    # --- PALABRAS CLAVE MEJORADAS ---
    
    # Vision: Añadimos 'merger' (Qwen Projector) y 'multi_modal_projector' (Gemma Projector)
    vision_keywords = [
        "visual", "vision_tower", "image_encoder", "patch_embed", 
        "multi_modal_projector", "merger" 
    ]
    
    # LLM: 'layers' es peligroso porque Gemma Vision tambien tiene 'layers'.
    # Usamos 'language_model' y 'text_model' como principales.
    llm_keywords = [
        "language_model", "text_model", "embed_tokens", "lm_head", 
        "layers." # Mantener al final como fallback
    ]

    vision_count = 0
    llm_count = 0
    other_count = 0

    for name in trainable_params:
        is_vision = any(k in name for k in vision_keywords)
        is_llm = any(k in name for k in llm_keywords)

        # LÓGICA DE PRIORIDAD
        # En Gemma 3, una capa puede llamarse 'vision_tower...layers...'.
        # Tiene keywords de ambos. Damos prioridad a VISION para que no cuente como LLM.
        if is_vision:
            vision_count += 1
        elif is_llm:
            llm_count += 1
        else:
            other_count += 1

    print("\n" + "-" * 60)
    print("REPORTE DE CAPAS ENTRENABLES (PEFT Check)")
    print("-" * 60)
    print(f"Total de tensores entrenables: {len(trainable_params)}")
    print(f" - Modulo de Vision (+Projector): {vision_count}")
    print(f" - Modulo de Lenguaje (LLM):      {llm_count}")
    print(f" - Otros/Indefinido:              {other_count}")
    print("-" * 60)
    
    print("CONCLUSION:")
    if vision_count > 0 and llm_count > 0:
        print("🚨 [MIXTO] Se estan entrenando simultaneamente Vision y Lenguaje.")
    elif vision_count > 0 and llm_count == 0:
        print("👁️  [SOLO VISION] El modelo de lenguaje (LLM) esta congelado.")
    elif llm_count > 0 and vision_count == 0:
        print("🧠 [SOLO LENGUAJE] El encoder visual esta congelado.")
    else:
        print("⚠️  [DESCONOCIDO] Revisa los nombres manualmente.")
    
    print("-" * 60)
    print("Muestra de las primeras 5 capas activas:")
    for name in trainable_params[:5]:
        print(f" > {name}")
    print("-" * 60 + "\n")


def load_model_and_processor(model_id, module, lora_r, lora_alpha, use_dora):
    """
    Load model and processor with PEFT configuration.
    
    Args:
        model_id: HuggingFace model ID
        module: Which module to train (all-linear, vision, language, CNN)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        use_dora: Whether to use DoRA
    """
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    if model_id != "google/gemma-3-12b-it":
        # Parche para evitar errores (aunque el Trainer estándar es más robusto que SFTTrainer)
        if not hasattr(processor, "pad_token"):
            print("WARNING: Original processor has no attribute called pad_token. Adding eos_token as processor pad_token.")
            processor.pad_token = processor.tokenizer.eos_token
    
    if model_id == "Qwen/Qwen3-VL-32B-Instruct":
        print("Loading Qwen base...")
        # Quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
            device_map="auto"
        )

        # Prepare the quantized model for training 
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    elif model_id == "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct":
        print("Loading LLaVA-OneVision-1.5-8B-Instruct base...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True
        )

    elif model_id == "google/gemma-3-12b-it":
        print("Loading Gemma base...")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
            device_map="auto"
        )

    else:
        raise ValueError(f"Tried to load invalid model: {model_id}")

    # Activar gradientes en entrada (para el warning de checkpointing)
    model.enable_input_require_grads()

    # Se deciden las capas que se entrenarán
    trainable_modules = []
    if module == "all-linear":
        trainable_modules = "all-linear"
    else:
        # (Aquí los nombres NO se repiten, basta con listas simples)
        if model_id in ["Qwen/Qwen3-VL-32B-Instruct", "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"]:
            if module == "vision":
                # Entrena Encoder Visual + Projector (Merger)
                trainable_modules = ["qkv", "proj", "linear_fc1", "linear_fc2"]
            elif module == "language":
                # Entrena solo el LLM
                trainable_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            elif module == "CNN":
                trainable_modules = ["model.visual.patch_embed.proj"]
        
        # (Aquí los nombres se repiten, ES OBLIGATORIO usar Regex para no mezclar)
        elif model_id == "google/gemma-3-12b-it":
            if module == "vision":
                # Busca capas dentro de 'vision_tower' (SigLIP)
                # Incluye q/k/v_proj (atención), out_proj (salida) y fc1/fc2 (MLP)
                trainable_modules = r".*vision_tower.*(q_proj|k_proj|v_proj|out_proj|fc1|fc2)$"
            
            elif module == "language":
                # Busca capas dentro de 'language_model' (Gemma)
                # Incluye q/k/v/o_proj (atención) y gate/up/down_proj (MLP)
                trainable_modules = r".*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        lora_dropout=0.1,
        use_dora=use_dora,
        bias="none",
        target_modules=trainable_modules
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    dtypes = Counter(param.dtype for param in model.parameters())
    print("Data type distributions:")
    for dtype, count in dtypes.items():
        print(f"- {dtype}: {count} tensors")

    return model, processor


def main():
    argparser = argparse.ArgumentParser(description="Train a visual language model with PEFT.")
    
    # Model and module selection
    argparser.add_argument(
        "--model-id", 
        type=str, 
        default="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        choices=[
            "Qwen/Qwen3-VL-32B-Instruct",
            "google/gemma-3-12b-it",
            "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
        ],
        help="HuggingFace model ID to use"
    )
    argparser.add_argument(
        "--module", 
        type=str, 
        default="all-linear",
        choices=["all-linear", "vision", "language", "CNN"],
        help="Module type to fine-tune"
    )
    
    # Dataset and output paths
    argparser.add_argument(
        "--dataset-path",
        type=str,
        default="/datafast/105-1/Datasets/INTERNS/aplanaj/hf_dataset_bev_ss_coord",
        help="Path to the dataset"
    )
    argparser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the trained model (default: auto-generated based on model and module)"
    )
    
    # Training hyperparameters
    argparser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    argparser.add_argument("--batch-size", type=int, default=1, help="Per-device training batch size")
    argparser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    argparser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    argparser.add_argument("--warmup-steps", type=int, default=50, help="Number of warmup steps")
    argparser.add_argument("--logging-steps", type=int, default=10, help="Logging frequency")
    
    # LoRA hyperparameters
    argparser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    argparser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha parameter")
    argparser.add_argument("--use-dora", action="store_true", help="Use DoRA instead of LoRA")
    
    # GPU selection
    argparser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')"
    )
    
    args = argparser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES if specified
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"Using GPUs: {args.cuda_visible_devices}")
    
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        model_name = args.model_id.split("/")[-1]
        args.output_dir = os.path.join(
            "/datafast/105-1/Datasets/INTERNS/aplanaj",
            f"{model_name}-lora-bev-ss-coord-{args.module.upper()}"
        )
    
    log_file = os.path.join(args.output_dir, "training_logs.jsonl")

    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {args.model_id}")
    print(f"Module: {args.module}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA r: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"DoRA: {args.use_dora}")
    print(f"{'='*60}\n")

    print("Loading model...")
    model, processor = load_model_and_processor(
        args.model_id, 
        args.module,
        args.lora_r,
        args.lora_alpha,
        args.use_dora
    )

    verify_trainable_layers(model)
    
    print("Loading dataset...")
    dataset = load_from_disk(args.dataset_path)
    
    # Configuración estándar (TrainingArguments en vez de SFTConfig)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False # Importante para que pase 'messages' e 'images' al collator
    )
    
    # Instanciar el Collator personalizado
    data_collator = DataCollatorForVisualSFT(args.model_id, processor)
    # Instanciar el Callback personalizado para logging (guardar datos de entrenamiento en tiempo real)
    logger_callback = LogLoggerCallback(log_path=log_file)
    
    # Usar el TRAINER estándar (no SFTTrainer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[logger_callback]
    )
    
    print("Starting training (Assistant Response Only)...")
    print(f"USING: {next(model.parameters()).device}")
    trainer.train()
    
    print(f"Saving model at {args.output_dir}...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()
