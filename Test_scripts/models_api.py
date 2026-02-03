import os
import torch
from PIL import Image
import math
#from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoTokenizer, AutoProcessor, AutoModelForCausalLM, pipeline, AutoModel, AutoModelForImageTextToText, BitsAndBytesConfig, TextStreamer
from transformers import Gemma3ForConditionalGeneration, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

from peft import PeftModel

import base64
import json
from google import genai
from google.genai import types
import io
import time
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../CILv2/CILv2_multiview/network/models/architectures/CIL_multiview"))
sys.path.append(root_path)
from CIL_multiview import CIL_multiview
from CIL_singleview import CIL_singleview

print(sys.path)

LORA_MODELS_BASE_DIRECTORY = os.path.join("/datafast", "105-1", "Datasets", "INTERNS", "aplanaj")

# ========================= LOAD LMM =========================

def load_lmm(model_name: str) -> tuple:
    """Carga un modelo de HuggingFace usando accelerate para modelos grandes."""

    print(f'Loading model: {model_name} with accelerate...')

    print(f"CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}")

    if model_name == "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct":
        return load_lmm_llava(model_name)
    elif model_name in ["google/gemma-3-12b-it", "google/gemma-3-27b-it"]:
        return load_lmm_gemma(model_name)
    elif model_name == "OpenGVLab/InternVL3_5-38B-HF":
        return load_lmm_intern(model_name)
    elif model_name.find("Qwen/Qwen3") != -1:
        return load_lmm_qwen(model_name)
    elif model_name == "google/gemini-2.5-flash":
        return load_lmm_gemini(model_name)
    elif model_name == "Qwen3-VL-32B-Instruct-lora":
        return load_lmm_qwen_lora(model_name)
    elif model_name.find("gemma-3-12b-it-lora") != -1:
        return load_lmm_gemma_lora(model_name)
    elif model_name.find("LLaVA-OneVision-1.5-8B-Instruct-lora") != -1:
        return load_lmm_llava_lora(model_name)
    elif model_name.find("Qwen3-VL-32B-Instruct-lora") != -1:
        return load_lmm_qwen_lora(model_name)
    elif model_name.find("CIL_customed") != -1:
        return load_lmm_cil(model_name)
    else:
        return None
    

def load_lmm_gemini(model_name: str) -> tuple:
    """ Load google/gemini-2.5-flash model. """
    return model_name, None, None, None


def load_lmm_llava(model_name: str) -> tuple:
    """ Load LLaVA model."""

    # Tokenizer: transforms text to tokens and viceversa
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Processor: process images (not all models have this)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    """
    # Config: load model architecture
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Model: load with empty weights first
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # Model: load weights and dispatch to devices. Distributes weights to GPUs automatically
    model = load_checkpoint_and_dispatch(
        model, 
        model_name, 
        device_map="auto"
    )
    """

    # Model: load with accelerate through AutoModelForCausalLM. load_checkpoint_and_dispatch could 
    # be also used but further configuration is needed
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",          # Automatically distribute layers across available GPUs
        trust_remote_code=True
    )

    model.eval()

    return (model_name, model, tokenizer, processor)



def load_lmm_gemma(model_name: str) -> tuple:
    """ Load Gemma models."""

    tokenizer = None # Not necessary

    processor = AutoProcessor.from_pretrained(model_name)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name, device_map="auto"
    ).eval()

    return (model_name, model, tokenizer, processor)



def load_lmm_intern(model_name: str) -> tuple:

    # Tokenizer: transforms text to tokens and viceversa
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Processor: process images (not all models have this)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        #use_flash_attn=False,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager"
    ).eval()

    return (model_name, model, tokenizer, processor)



def load_lmm_qwen(model_name: str) -> tuple:

    # Tokenizer: does not use
    tokenizer = None

    # Processor: process images (not all models have this)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if model_name == "Qwen/Qwen3-VL-235B-A22B-Instruct":

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            trust_remote_code=True,
            device_map="auto",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).eval()   

        # # Quantization configuration
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )

        # max_memory_mapping = {
        #     0: "12GiB",
        #     1: "12GiB",
        #     2: "12GiB",
        #     3: "12GiB",
        #     4: "12GiB",
        #     5: "12GiB",
        #     6: "12GiB",
        #     7: "42GiB",
        # }

        # model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        #     model_name,
        #     attn_implementation="sdpa",
        #     trust_remote_code=True,
        #     device_map="auto",
        #     max_memory=max_memory_mapping,
        #     quantization_config=bnb_config,
        #     low_cpu_mem_usage=True
        # ).eval()

    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
            device_map="auto"
        ).eval()

    return (model_name, model, tokenizer, processor)



def load_lmm_qwen_lora(model_name: str) -> tuple:

    print(f"Loading LoRA fine-tuned {model_name} model...")

    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-32B-Instruct",
        dtype="auto",
        attn_implementation="eager",
        device_map="auto",
        quantization_config=bnb_config
    ).eval()
    
    model = PeftModel.from_pretrained(base, os.path.join(LORA_MODELS_BASE_DIRECTORY, model_name))

    processor = AutoProcessor.from_pretrained(os.path.join(LORA_MODELS_BASE_DIRECTORY, model_name), trust_remote_code=True)

    return (model_name, model, None, processor)



def load_lmm_gemma_lora(model_name: str) -> tuple:

    print(f"Loading LoRA fine-tuned {model_name} model...")

    base = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-12b-it",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    model = PeftModel.from_pretrained(base, os.path.join(LORA_MODELS_BASE_DIRECTORY, model_name))

    #processor = AutoProcessor.from_pretrained(os.path.join(LORA_MODELS_BASE_DIRECTORY, model_name), trust_remote_code=True)

    processor = AutoProcessor.from_pretrained("google/gemma-3-12b-it", trust_remote_code=True)

    return (model_name, model, None, processor)



def load_lmm_llava_lora(model_name) -> tuple:

    print(f"Loading LoRA fine-tuned {model_name} model...")

    base =  AutoModelForCausalLM.from_pretrained(
        "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        device_map="auto",
        trust_remote_code=True
    ).eval()

    model = PeftModel.from_pretrained(base, os.path.join(LORA_MODELS_BASE_DIRECTORY, model_name))

    processor = AutoProcessor.from_pretrained(os.path.join(LORA_MODELS_BASE_DIRECTORY, model_name), trust_remote_code=True)

    return (model_name, model, None, processor)  



def load_lmm_cil(model_name: str) -> tuple:

    print(f"Loading CIL {model_name} model...")

    model = torch.load(os.path.join(LORA_MODELS_BASE_DIRECTORY, model_name + '.pth'), weights_only=False)
    model.eval()

    return (model_name, model, None, None)


# ========================= CALL LMM =========================


def call_lmm(pipe_elements: tuple, prompt: str, images: any, icl_enabled: bool = False, annotations_path: str = None, annotation_key: str = None, front_img_type: str = "rgb", icl_style: str = "v2") -> str:
    """
    Función principal para llamar a los modelos.
    
    :param images: Puede ser una lista de [Image.Image] (comportamiento antiguo) 
                   o un diccionario {'front_imgs_rgb': PIL, 'front_imgs_ss': PIL, 'map_img_rgb': PIL} (nuevo comportamiento).
    :param front_img_type: 'rgb', 'ss' o None. Define qué imagen frontal usar.

    :param icl_style: 'v1' / 'v2'. Define qué versión de ICL usar.
    """
    model_name = pipe_elements[0]

    # --- 1. Pre-procesamiento de imágenes (Selección de Frontal/Mapa) ---
    final_images_list = []
    
    if isinstance(images, dict):
        # Seleccionar imagen frontal
        if front_img_type == "rgb" and "front_imgs_rgb" in images:
            final_images_list.append(images["front_imgs_rgb"])
        elif front_img_type == "ss" and "front_imgs_ss" in images:
            final_images_list.append(images["front_imgs_ss"])
        elif front_img_type is None:
            pass 
            
        # Seleccionar imagen de mapa
        if "map_img_rgb" in images:
            raise ValueError("BEV RGB view was requested.")
    elif isinstance(images, list):
        final_images_list = images
    else:
        return "Error: Formato de imágenes no reconocido."

    # --- 2. Enrutamiento ICL (In-Context Learning) ---
    if icl_enabled and annotations_path:
        if icl_style == "v1":
            return call_lmm_icl(pipe_elements, prompt, final_images_list, annotations_path, annotation_key, front_img_type)
        elif icl_style == "v2":
            return call_lmm_icl_v2(pipe_elements, prompt, final_images_list, annotations_path, annotation_key, front_img_type)
        else:
            raise ValueError("Incorrect icl_style parameter value. Use 'v1' or 'v2'.")

    # --- 3. Llamadas Estándar (Sin ICL) ---
    # (El resto de la función se mantiene igual que tu original...)
    if model_name == "OpenGVLab/InternVL3_5-38B-HF" or model_name.find("LLaVA-OneVision-1.5-8B-Instruct") != -1:
        return call_lmm_llava(pipe_elements, prompt, final_images_list)
    elif model_name.find("gemma-3") != -1: 
        return call_lmm_gemma(pipe_elements, prompt, final_images_list)
    elif model_name.find("Qwen3-VL") != -1: 
        return call_lmm_qwen(pipe_elements, prompt, final_images_list)
    elif model_name == "google/gemini-2.5-flash":
        return call_lmm_gemini(pipe_elements, prompt, final_images_list)
    elif model_name.find("CIL_customed") != -1:
        return call_lmm_cil(pipe_elements, prompt, final_images_list)
    else:
        return f"Error: Modelo {model_name} no reconocido en call_lmm."

# ========================= ICL V2 (NEW METHOD) =========================

def call_lmm_icl_v2(pipe_elements: tuple, prompt: str, task_images: [Image.Image], annotations_path: str, annotation_key: str, front_img_type: str) -> str:
    """
    Implementación alternativa de ICL basada en la lógica Qwen3-VL (Reference Script).
    
    Diferencias clave con v1:
    1. Orden: Ejemplos primero -> Tarea actual después.
    2. Gestión de imágenes: Lista plana 'all_images' sincronizada.
    3. Procesamiento: apply_chat_template(tokenize=False) -> processor(text=text, images=all_images).
    """
    model_name, model, tokenizer, processor = pipe_elements
    print(f"--- Running ICL V2 (Sequence: Examples -> Task) for {model_name} ---")

    extended_prompt =r"""NOTE: Based on the direction of the red arrow and the blue square position, determine **FROM THE DRIVER'S PERSPECTIVE** if I have to continue Straight, turn Right or turn Left to reach the blue square.
It is crucial that you correctly identify **THE DIRECTION THE ARROW POINTS TO AND THE RELATIVE POSITION OF THE BLUE SQUARE** to provide an accurate answer.

NOTE: You could get stuck in a loop while reasoning. Be careful to avoid this situation by not repeating the same reasoning steps. If somethings seems to be too complicated (maybe a U-turn), just provide the most probable direction."

FORMAT: 
1. You can provide your own analysis, reasoning or thoughts that guide you to the answer between <reasoning> and </reasoning> tags.
2. Provide the final answer as exactly one of these three words: 'Straight', 'Right', or 'Left'. Provide **ONLY ONE WORD** after the </reasoning> tag."""

    prompt += "\n\n" + extended_prompt

    try:
        # 1. Cargar anotaciones
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # Listas para mantener sincronización estricta (Lógica del script referencia)
        all_images = []
        message_content = []

        # --- PARTE A: Insertar Ejemplos (Few-Shot) PRIMERO ---
        header_text = "Here are some examples of how to analyze these situations:\n\n"
        message_content.append({"type": "text", "text": header_text})

        # Determinar clave para imagen frontal
        front_key = "front_imgs_ss" if front_img_type == "ss" else "front_imgs_rgb"
        use_front = (front_img_type is not None)

        for idx, annotation in enumerate(annotations): 
            try:
                message_content.append({"type": "text", "text": f"Example {idx+1}\n"})

                # 1. Imagen Frontal del ejemplo (si aplica)
                if use_front:
                    if front_key in annotation:
                        ex_front_img = Image.open(annotation[front_key])
                        all_images.append(ex_front_img)
                        message_content.append({"type": "image", "image": ex_front_img})
                    else:
                        print(f"Warning ICL V2: Missing {front_key} in annotation {idx}")

                # 2. Imagen BEV del ejemplo
                ex_map_img = Image.open(annotation["map_img_ss"])
                all_images.append(ex_map_img)
                message_content.append({"type": "image", "image": ex_map_img})

                # 3. Solución (Ground Truth)
                gt_text = annotation.get("annotations", {}).get(annotation_key, "")
                if not gt_text:
                    # Intento de fallback a estructura plana si 'annotations' key no existe
                    gt_text = annotation.get(annotation_key, "No solution provided")
                
                message_content.append({"type": "text", "text": f"\n{gt_text}\n"})

            except Exception as e:
                print(f"Skipping ICL V2 example {idx} due to error: {e}")

        # --- PARTE B: Insertar la Tarea Actual (Query) AL FINAL ---
        message_content.append({"type": "text", "text": "\nCURRENT TASK:\n"})
        
        # Añadir las imágenes de la tarea actual a la lista global y al contenido
        for img in task_images:
            all_images.append(img)
            message_content.append({"type": "image", "image": img})
        
        # Añadir el prompt de la tarea
        message_content.append({"type": "text", "text": prompt})

        # --- PROCESAMIENTO ---
        
        # Caso especial para Gemini (API)
        if model_name == "google/gemini-2.5-flash":
            # Reutilizamos el helper de Gemini, ya que message_content tiene la estructura correcta
            return _infer_gemini_icl(message_content)

        # Caso Modelos HuggingFace (Qwen, Gemma, LLaVA)
        
        # 0. Crea el Streamer para visualizar la salida en tiempo real
        streamer = TextStreamer(processor.tokenizer, skip_prompt=False, skip_special_tokens=True)

        # 1. Preparar lista de mensajes
        messages = [{"role": "user", "content": message_content}]

        # 2. Generar TEXTO del prompt (Sin tokenizar) -> CLAVE DEL NUEVO MÉTODO
        text_prompt = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 3. Pasar texto e imágenes por separado al processor
        inputs = processor(
            text=[text_prompt],
            images=all_images,
            padding=True,
            return_tensors="pt"
        )

        # Mover a GPU
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print(f"V2 Inputs prepared. Image count: {len(all_images)}. Tokenizing complete.")

        # 4. Generación
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=8192, 
                do_sample=False
            )

        # 5. Decodificación (Recortando input)
        if "Qwen" in model_name:
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        else:
            # Lógica genérica para otros modelos si soportan este flujo
            input_len = inputs["input_ids"].shape[-1]
            output_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True)

        print(f"LMM Response (ICL V2): {output_text}")
        return output_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error V2: {str(e)}"



# ========================= ICL UNIFIED FUNCTION =========================

def call_lmm_icl(pipe_elements: tuple, prompt: str, task_images: [Image.Image], annotations_path: str, annotation_key: str, front_img_type: str) -> str:
    """
    Función unificada para In-Context Learning.
    Carga ejemplos del JSON, construye el historial de chat con imágenes intercaladas
    y llama a la inferencia específica del modelo.
    """
    model_name, model, tokenizer, processor = pipe_elements
    
    print(f"--- Running ICL for {model_name} (Front type: {front_img_type}) ---")

    try:
        # 1. Cargar anotaciones
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        # 2. Construir la estructura del mensaje (Formato 'content' lista mixta)
        # Esto es compatible con Qwen, Gemma y LLaVA (vía apply_chat_template)
        message_content = []

        # A. Instrucción inicial
        header_text = prompt + "\n\nHere are some examples to help you:\n"
        message_content.append({"type": "text", "text": header_text})

        # B. Iterar ejemplos
        for idx, annotation in enumerate(annotations):
            try:
                # Determinar qué key usar del JSON para la imagen FRONTAL (RBG/SS) según el parámetro
                front_key = "front_imgs_ss" if front_img_type == "ss" else "front_imgs_rgb"
                
                # Si se pidió 'None' (sin frontal), en los ejemplos ICL tampoco ponemos frontal
                use_front = (front_img_type is not None)

                # Cargar imágenes del ejemplo
                map_img = Image.open(annotation["map_img_ss"])
                
                # Añadir imágenes al contenido
                if use_front:
                    if front_key in annotation:
                        front_img = Image.open(annotation[front_key])
                        message_content.append({"type": "image", "image": front_img})
                    else:
                        print(f"Warning ICL: Missing {front_key} in annotation {idx}")
                
                message_content.append({"type": "image", "image": map_img})

                # Añadir texto de respuesta correcta (Ground Truth)
                # Usamos 'annotation' si existe, sino construimos "Answer: ..."
                gt_text = annotation.get("annotations", None).get(annotation_key, None)
                if not gt_text:
                    raise ValueError("Annotation not found")
                message_content.append({"type": "text", "text": f"\nExample {idx+1} Solution:\n{gt_text}\n\n"})

            except Exception as e:
                print(f"Skipping ICL example {idx} due to error: {e}")
                continue

        # C. El caso real a resolver (Task)
        message_content.append({"type": "text", "text": "Now, it is your turn to solve the following scenario:\n"})
        for img in task_images:
            message_content.append({"type": "image", "image": img})

        print("FULL ICL MESSAGE:\n",message_content)
        
        # --- 3. Inferencia según el modelo ---
        
        # >>>> GEMINI (API) <<<<
        if model_name == "google/gemini-2.5-flash":
            return _infer_gemini_icl(message_content)

        # >>>> MODELOS HUGGINGFACE (Qwen, Gemma, LLaVA) <<<<
        
        # Preparar mensaje para chat template
        conversation = [{"role": "user", "content": message_content}]
        
        # Procesar inputs
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        print("AFTER APPLYING CHAT TEMPLATE:", processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))
        print("TOTAL NUMBER OF TOKENS: ", inputs["input_ids"].shape)

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generar
        # Ajustes específicos por familia de modelos
        max_tokens = 1024 if "Qwen" in model_name else 4096
        
        with torch.no_grad(): # o torch.inference_mode()
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

        # Decodificar
        if "Qwen" in model_name:
            # Qwen devuelve input+output, hay que recortar
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        elif "gemma" in model_name or "LLaVA" in model_name:
            # Gemma/LLaVA a veces devuelven todo, a veces solo lo nuevo, depende de la config.
            # Generalmente recortamos por longitud de input para estar seguros.
            input_len = inputs["input_ids"].shape[-1]
            generated_ids_trimmed = generated_ids[0][input_len:]
            output_text = processor.decode(generated_ids_trimmed, skip_special_tokens=True)
        else:
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"LMM Response (ICL): {output_text}")
        return output_text

    except Exception as e:
        print(f"Error in ICL execution: {str(e)}")
        return f"Error: {str(e)}"

def _infer_gemini_icl(message_content):
    """ Helper privado para convertir el formato de mensajes a Gemini API Types """
    try:
        parts = []
        for item in message_content:
            if item["type"] == "text":
                parts.append(types.Part.from_text(text=item["text"]))
            elif item["type"] == "image":
                buffer = io.BytesIO()
                item["image"].save(buffer, format="PNG")
                parts.append(types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"))
        
        contents = [types.Content(role="user", parts=parts)]
        
        client = genai.Client(api_key="YOUR API KEY")
        
        print("CALLING GEMINI API (ICL mode)...")
        
        config = types.GenerateContentConfig(
            top_p=1,
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            image_config=types.ImageConfig(image_size="1K")
        )

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash", contents=contents, config=config
        ):
            print(chunk.text, end="")
            response_text += chunk.text
            time.sleep(1) # Pequeña pausa para no saturar consola
        
        time.sleep(5)
        return response_text

    except Exception as e:
        return f"Error Gemini ICL: {str(e)}"


# ========================= STANDARD CALL FUNCTIONS (NO CHANGE OR MINOR ADAPTS) =========================

def call_lmm_llava(pipe_elements: tuple, prompt: str, images: [Image.Image]) -> str:
    """ Call LLaVA standard. """
    model_name, model, tokenizer, processor = pipe_elements
    print(f"Answering {model_name}\n")
    try:
        message = [{"role": "user", "content": []}]
        for image in images:
            message[0]["content"].append({"type": "image", "image": image})
        message[0]["content"].append({"type": "text", "text": prompt})

        inputs = processor.apply_chat_template(
            conversation=message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        
        text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # A veces LLaVA repite el prompt, limpiamos si es necesario (simple split si el modelo lo requiere)
        if "assistant" in text: 
             text = text.split("assistant")[-1].strip()
             
        print(f"LMM response: {text}")
        return text
    except Exception as e:
        print("Error:", str(e))
        return f"Error: {str(e)}"


def call_lmm_gemma(pipe_elements: tuple, prompt: str, images: [Image.Image]) -> str:
    """ Call Gemma standard. """
    model_name, model, tokenizer, processor = pipe_elements
    try:
        message = [{"role": "user", "content": []}]
        for image in images:
            message[0]["content"].append({"type": "image", "image": image})
        message[0]["content"].append({"type": "text", "text": prompt})

        inputs = processor.apply_chat_template(
            conversation=message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
            generation = generation[0][input_len:] # Solo la parte generada

        decoded = processor.decode(generation, skip_special_tokens=True)
        print(f"LMM response: {decoded}")
        return decoded
    except Exception as e:
        print("Error:", str(e))
        return f"Error: {str(e)}"


def call_lmm_qwen(pipe_elements: tuple, prompt: str, images: [Image.Image]) -> str:
    """ Call Qwen standard. """
    model_name, model, tokenizer, processor = pipe_elements

    if model_name == "Qwen3-VL-32B-Thinking":
        prompt = prompt + "\nNOTE: You could get stuck in a loop while reasoning. Be careful to avoid this situation by not repeating the same reasoning steps. If somethings seems to be too complicated, just stop thinking and provide an answer."

    try:
        message = [{"role": "user", "content": []}]
        for image in images:
            message[0]["content"].append({"type": "image", "image": image})
        message[0]["content"].append({"type": "text", "text": prompt})

        inputs = processor.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(f"LMM response: {output_text}")
        return output_text
    except Exception as e:
        print("Error:", str(e))
        return f"Error: {str(e)}"


def call_lmm_qwen_lora(pipe_elements, prompt, images: [Image.Image]) -> str:
    # Redirige al estándar de Qwen, la lógica es la misma
    return call_lmm_qwen(pipe_elements, prompt, images)

def call_lmm_gemma_lora(pipe_elements, prompt, images: [Image.Image]) -> str:
    # Redirige al estándar de Gemma
    return call_lmm_gemma(pipe_elements, prompt, images)


def call_lmm_gemini(pipe_elements: tuple, prompt: str, images: [Image.Image]) -> str:
    """ Call Gemini Standard (No ICL logic here). """
    try:
        def pil_to_part(img):
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")

        image_parts = [pil_to_part(image) for image in images]
        
        # En Gemini la API permite el prompt como system instruction o como parte del user content.
        # Aquí lo mantenemos en system instruction como en tu código original, o en user.
        # Lo pongo en user para consistencia con ICL.
        
        client = genai.Client(api_key="YOUR API KEY")
        
        # Contenido: Prompt texto + Imágenes
        user_parts = [types.Part.from_text(text=prompt)] + image_parts
        
        contents = [types.Content(role="user", parts=user_parts)]

        print("CALLING GEMINI API\n")
        generate_content_config = types.GenerateContentConfig(
            top_p=1,
            thinking_config = types.ThinkingConfig(thinking_budget=-1),
            image_config=types.ImageConfig(image_size="1K"),
        )

        full_text = ""
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=contents,
            config=generate_content_config,
        ):
            print(chunk.text, end="")
            full_text += chunk.text
            time.sleep(1)

        time.sleep(30)
        return full_text
    
    except Exception as e:
        print("Error:", str(e))
        time.sleep(35)
        return f"Error: {str(e)}"


def call_lmm_cil(pipe_elements, prompt, images):
    _, model, _, _ = pipe_elements
    with torch.no_grad():
        # CIL espera 2 imágenes específicas
        if len(images) == 2:
            output = model.inference_from_pil(images[0], images[1])
        elif len(images) == 1:
            output = model.inference_from_pil(images[0]) 
        else:
            return "Error: CIL requiere 1 (singleview) o 2 (multiview) imágenes."
    return output