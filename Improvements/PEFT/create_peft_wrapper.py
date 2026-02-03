import argparse
from transformers import AutoProcessor
from datasets import load_from_disk
import os


# --- 1. LA LÓGICA DE ENMASCARAMIENTO ---

def find_assistant_start(input_ids, processor):
    """
    Busca dónde empieza la respuesta del asistente para enmascarar lo anterior.
    En Qwen, la respuesta empieza después de: <|im_start|>assistant\n
    """
    # Estos IDs dependen del tokenizer de Qwen. 
    # Generalmente es la secuencia: [im_start_id, assistant_id]
    # Lo buscamos dinámicamente para no hardcodear números.
    
    im_start_id = processor.tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
    assistant_id = processor.tokenizer.encode("assistant", add_special_tokens=False)[0]
    
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
    def __init__(self, processor):
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
                add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["images"])

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            max_length=1024,
            truncation=False,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        
        # Máscara de padding estándar
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        for i in range(len(labels)):
            # Usamos tu función find_assistant_start
            # Asegúrate de que retorna el índice del PRIMER token de la respuesta (la palabra)
            start_idx = find_assistant_start(batch["input_ids"][i], self.processor)
            
            # Verificación de seguridad (Assert mejorado)
            token_id = batch["input_ids"][i, start_idx].item()
            if token_id not in self.valid_ids:
                decoded = self.tokenizer.decode([token_id])
                # Lanzamos error con información útil
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



def main():
    parser = argparse.ArgumentParser(description="Test PEFT wrapper with a specific model and dataset.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        help="HuggingFace model ID to use"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/datafast/105-1/Datasets/INTERNS/aplanaj/hf_dataset_ss_coord",
        help="Path to the dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to test"
    )
    
    args = parser.parse_args()

    print(f"Loading processor for model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    muestra = [dataset[i] for i in range(args.num_samples)]
    print(muestra,"\n\n")

    enc = processor.apply_chat_template(
                muestra[0]["messages"], 
                tokenize=False,
                add_special_tokens=False,
                add_generation_prompt=False # Always set to False, as the conversational format already includes the generation prompt
    )

    enc = processor(
        text = enc,
        images = muestra[0]["images"],
        truncation = False,
        add_special_tokens=False,
        max_length=1024,
        return_tensors="pt"
    )

    print("Shape inputs_ids: ", enc["input_ids"].shape)

    for token_id in enc["input_ids"][0]:
        print(f"Token ID: {token_id} --> Símbolo: {processor.tokenizer.decode(token_id)}")


if __name__=="__main__":
    main()
