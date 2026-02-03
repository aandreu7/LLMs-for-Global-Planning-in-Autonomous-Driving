import os
"""
os.environ["CUDA_PATH"] = "/usr/local/cuda-11.4"
os.environ["PATH"] = f"/usr/local/cuda-11.4/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/usr/local/cuda-11.4/lib64"
os.environ["CUDA_HOME"] = "/usr/local/cuda-11.4"
os.environ["CUDA_BIN_PATH"] = "/usr/local/cuda-11.4/bin"
os.environ["CUDA_TOOLKIT_ROOT_DIR"] = "/usr/local/cuda-11.4"
"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import json
import re
import time
import argparse
from typing import Any, List, Dict
from PIL import Image
import numpy as np
import pickle

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score

# Asegúrate de importar call_lmm y load_lmm
from models_api import * # ========================================================================================================
# HELPER FUNCTIONS
# ========================================================================================================

def get_balanced_subset(dataset: List[Dict[str, Any]], num_samples: int, random_seed: int = None) -> List[Dict[str, Any]]:
    if random_seed is not None:
        np.random.seed(random_seed)
    
    samples_per_class = num_samples // 3
    classes = {"Straight": [], "Right": [], "Left": []}
    
    for sample in dataset:
        correct_exit = sample.get("ground_truth", {}).get("correct_exit")
        if correct_exit in classes:
            classes[correct_exit].append(sample)
    
    for class_name, samples in classes.items():
        if len(samples) < samples_per_class:
            raise ValueError(f"No hay suficientes muestras de '{class_name}'. Disp: {len(samples)}, Nec: {samples_per_class}.")
    
    balanced_subset = []
    for class_name in ["Straight", "Right", "Left"]:
        indices = np.random.choice(len(classes[class_name]), samples_per_class, replace=False)
        selected_samples = [classes[class_name][i] for i in indices]
        balanced_subset.extend(selected_samples)
    
    np.random.shuffle(balanced_subset)
    counts = {"Straight": 0, "Right": 0, "Left": 0}
    for sample in balanced_subset:
        counts[sample["ground_truth"]["correct_exit"]] += 1
    print(f"Distribución: {counts}")
    return balanced_subset

def split_dataset(dataset):
    train, validation = [], []
    for sample in dataset:
        if not sample.get("checked") or sample.get("skip") or not sample.get("clean_bev_ss_image"):
            continue
        if sample["clean_bev_ss_image"].find("Town01") != -1:
            train.append(sample)
        elif sample["clean_bev_ss_image"].find("Town02") != -1:
            validation.append(sample)
        else:
            print("!!! WARNING: INVALID MAP FOUND WHILE SPLITTING DATASET !!!")
    return train, validation

def show_test_results(y_test: list, y_pred_rgb: list, y_pred_ss: list, model_name: str, test_name: str):
    y_pred_rgb = [0 for i in range(len(y_pred_ss))] # QUITAR ESTO
    for vals in zip(y_test, y_pred_rgb, y_pred_ss):
        if any(v not in [0, 1, 2] for v in vals):
             print(f"!!! WARNING !!! INVALID CLASS FOUND in {vals}")

    acc_rgb = accuracy_score(y_test, y_pred_rgb)
    acc_ss = accuracy_score(y_test, y_pred_ss)
    f1_rgb = f1_score(y_test, y_pred_rgb, average='macro')
    f1_ss = f1_score(y_test, y_pred_ss, average='macro')
    precision_rgb = precision_score(y_test, y_pred_rgb, average='macro')
    precision_ss = precision_score(y_test, y_pred_ss, average='macro')
    recall_rgb = recall_score(y_test, y_pred_rgb, average='macro')
    recall_ss = recall_score(y_test, y_pred_ss, average='macro')

    conf_mat_rgb = confusion_matrix(y_test, y_pred_rgb)
    conf_mat_ss = confusion_matrix(y_test, y_pred_ss)

    save_dir = f"../carla_test/test_results/{model_name}/{test_name}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Accuracy RGB: {acc_rgb}")
    print(f"Accuracy SS: {acc_ss}")
    print(f"F1_macro RGB: {f1_rgb}")
    print(f"F1_macro SS: {f1_ss}")
    print(f"Confusion matrix RGB:\n{conf_mat_rgb}")
    print(f"Confusion matrix SS:\n{conf_mat_ss}")

    return acc_rgb, acc_ss, f1_rgb, f1_ss, precision_rgb, precision_ss, recall_rgb, recall_ss, conf_mat_rgb.tolist(), conf_mat_ss.tolist()

def show_test_results_gm(y_test: list, y_pred: list, model_name: str, test_name: str):
    acc_gm = accuracy_score(y_test, y_pred)
    f1_gm = f1_score(y_test, y_pred, average='macro')
    precision_gm = precision_score(y_test, y_pred, average='macro')
    recall_gm = recall_score(y_test, y_pred, average='macro')
    conf_mat_gm = confusion_matrix(y_test, y_pred)

    print("\n=== RESULTS GM ONLY ===")
    print(f"Accuracy:  {acc_gm:.3f}")
    print(f"F1 Score:  {f1_gm:.3f}")
    print("Confusion Matrix:\n", conf_mat_gm)

    save_dir = f"../carla_test/test_results/{model_name}/{test_name}"
    os.makedirs(save_dir, exist_ok=True)
    return acc_gm, f1_gm, precision_gm, recall_gm, conf_mat_gm.tolist()

def action2number(action: str) -> int:
    if not action: return -1
    action_normal = re.sub(r"[\s_-]+", "", action.lower().strip())
    if action_normal in ("straight", "lanefollow"): return 0
    if action_normal in ("right", "changelaneright"): return 1
    if action_normal in ("left", "changelaneleft"): return 2
    return -1

def normalize_lmm_choice(s: str) -> str:
    if s is None: return ''
    s = s.strip().lower().strip('. ,;:')
    s = s.replace('to the right', 'right').replace('to the left', 'left')
    s = s.replace('straight on', 'straight').replace('go straight', 'straight').replace('forward', 'straight')
    s = re.sub(r"[\s_-]+", "", s)
    mapping = {'right': 'Right', 'left': 'Left', 'straight': 'Straight', 'lanefollow': 'LaneFollow'}
    return mapping.get(s, s)

def parse_llm_response(response: str) -> dict[str, Any]:
    res = {'choice': "", 'justification': "", 'raw_all': response}
    if not response: return res
    
    # 1. Limpieza básica y eliminación de tags conocidos
    text = response.strip().replace('\r', '\n')
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Limpieza de separadores de chat comunes
    parts = re.split(r'assistant', text, flags=re.IGNORECASE)
    final_answer_text = parts[-1].strip() if len(parts) > 1 else text
    
    lines = [l.strip() for l in final_answer_text.splitlines() if l.strip()]
    valid_choices = ['straight', 'right', 'left']
    
    choice_line = None

    # ESTRATEGIA 1: Buscar coincidencia EXACTA desde el FINAL hacia arriba
    # Esto es crucial para modelos CoT que concluyen al final.
    for line in reversed(lines):
        clean = line.lower().strip('. ,;:!?"\'*')
        if clean in valid_choices:
            choice_line = clean
            break
            
    # ESTRATEGIA 2: Si no hay exacta, buscar línea CORTA que contenga la palabra clave (desde el final)
    if not choice_line:
        for line in reversed(lines):
            clean = line.lower().strip()
            # Si la línea es muy larga (>50 chars), es razonamiento, la ignoramos aunque tenga "Left"
            if len(clean) > 50: 
                continue
                
            found = False
            for v in valid_choices:
                # Buscamos la palabra como palabra completa o al final
                if v in clean:
                    choice_line = v
                    found = True
                    break
            if found: break

    res['choice'] = normalize_lmm_choice(choice_line) if choice_line else ""

    return res


def _save_wrong_classified(model_name: str, test_name: str, test_list: list, y_test: list, preds: dict):
    """Guarda en ./wrong_classified/<model_name>/<test_name>/ todas las imágenes
    de los casos que el modelo ha clasificado incorrectamente.

    preds: diccionario con listas de predicciones, por ejemplo {'rgb': [...], 'ss': [...]}
    Las listas en preds deben corresponder por índice con y_test y con los tests
    filtrados (solo aquellos usados en la evaluación, normalmente los que tienen
    `not test.get('real_image')`).
    """
    base_dir = os.path.join("..", "carla_test", "wrong_classified")
    # Si el script se ejecuta con cwd == CARLA_scripts, usar ./wrong_classified
    if os.path.isdir("./wrong_classified"):
        base_dir = os.path.join(".", "wrong_classified")

    model_dir = os.path.join(base_dir, model_name)
    target_dir = os.path.join(model_dir, test_name)
    os.makedirs(target_dir, exist_ok=True)

    # Filtrar tests en los que se generaron predicciones (misma lógica que en los tests)
    filtered_tests = [t for t in test_list if not t.get("real_image", None)]

    for idx, gt in enumerate(y_test):
        for pred_key, pred_list in preds.items():
            if idx >= len(pred_list):
                continue
            pred = pred_list[idx]
            if pred != gt:
                case_dir = os.path.join(target_dir, f"case_{idx}")
                os.makedirs(case_dir, exist_ok=True)

                # Guardar metadatos
                meta_path = os.path.join(case_dir, "info.txt")
                try:
                    with open(meta_path, 'w', encoding='utf-8') as mf:
                        mf.write(f"ground_truth: {gt}\n")
                        mf.write(f"prediction_{pred_key}: {pred}\n")
                except Exception:
                    pass

                # Intentar guardar imágenes relevantes si existen en el dict del test
                test = filtered_tests[idx] if idx < len(filtered_tests) else None
                if not test:
                    continue
                # keys posibles: map_img_ss, map_img_rgb, front_imgs_ss, front_imgs_rgb
                for key in ("map_img_ss", "map_img_rgb", "front_imgs_ss", "front_imgs_rgb"):
                    if key in test and test[key]:
                        try:
                            img = Image.open(test[key])
                            filename = f"{key}_gt{gt}_pred{pred}_{pred_key}.png"
                            img.save(os.path.join(case_dir, filename))
                        except Exception:
                            # Si el valor ya es un PIL Image
                            try:
                                if isinstance(test[key], Image.Image):
                                    img = test[key]
                                    filename = f"{key}_gt{gt}_pred{pred}_{pred_key}.png"
                                    img.save(os.path.join(case_dir, filename))
                            except Exception:
                                continue


def apply_branch_classifier(sample: dict) -> list:
    """
    0 --> ["Straight", "Right"]
    1 --> ["Straight", "Left"]
    2 --> ["Right", "Left"]
    """

    ss_pil = Image.open(sample["front_imgs_ss"]).convert("RGB")
    ss_np = np.array(ss_pil)
    ss_np = ss_np[100:200,:,:]
    binary_ss = np.zeros(ss_np.shape[:2], dtype=np.uint8)
    
    colors = [[128, 64, 128]]

    for color in colors:
        match = np.all(ss_np == color, axis=-1)
        binary_ss[match] = 255

    Image.fromarray(binary_ss).save("binarized.png")
    binary_ss = binary_ss.reshape(1, -1)

    output = BRANCH_CLASSIFIER.predict(binary_ss)

    output = ["Straight", "Right"] if output==0 else ["Straight", "Left"] if output==1 else ["Right", "Left"] if output==2 else -1

    assert output!=-1, "Invalid classification made by Branch Classifier"

    print(f"Branch Classifier predicts: {output}")

    return output

# ========================================================================================================
# TEST FUNCTIONS
# ========================================================================================================

def test_bev(test_list: list, pipe_elements: tuple, all_parsed_answers: dict, use_icl=False, use_bc=False, annotations_path=None, annotation_key=None) -> dict:
    all_parsed_answers["test_bev"] = {}
    y_pred_rgb, y_pred_ss, y_test = [], [], []
    total_inference_time = 0

    for i, test in enumerate(test_list):
        if not test.get("real_image", None):
            all_parsed_answers["test_bev"][f"test_case_{i}"] = {}

            prompt = PROMPT_TEST_BEV

            if use_bc:
                possible_branches = apply_branch_classifier(test)
                prompt = prompt + f"\n**HINT**: Available options in this scenario are {possible_branches[0]} and {possible_branches[1]}."

                print(prompt)

            # ===== SS TURN =====
            start_time = time.time()
            raw_answer = call_lmm(pipe_elements, prompt, images=[Image.open(test["map_img_ss"])], icl_enabled=use_icl, annotations_path=annotations_path, annotation_key="annotation_bev", front_img_type=None)
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            parsed_answer = parse_llm_response(raw_answer)
            all_parsed_answers["test_bev"][f"test_case_{i}"]["ss"] = parsed_answer
            actionAsNumber = action2number(parsed_answer["choice"])

            while actionAsNumber == -1:
                time.sleep(1)
                raw_answer = call_lmm(pipe_elements, prompt, images=[Image.open(test["map_img_ss"])], icl_enabled=use_icl, annotations_path=annotations_path, annotation_key="annotation_bev", front_img_type=None)
                parsed_answer = parse_llm_response(raw_answer)
                all_parsed_answers["test_bev"][f"test_case_{i}"]["ss"] = parsed_answer
                actionAsNumber = action2number(parsed_answer["choice"])
            y_pred_ss.append(actionAsNumber)

            y_test.append(action2number(test["ground_truth"]["correct_exit"]))

    if len(test_list) > 0: total_inference_time /= (len(test_list) * 2)
    print("Finished test_bev.")
    # Guardar imágenes mal clasificadas
    try:
        _save_wrong_classified(pipe_elements[0], "test_bev", test_list, y_test, {"rgb": y_pred_rgb, "ss": y_pred_ss})
    except Exception as e:
        print(f"Error saving wrong classified (test_bev): {e}")
    return _pack_results(y_test, y_pred_rgb, y_pred_ss, total_inference_time, pipe_elements[0], "test_bev")

def test_bev_frontal(test_list: list, pipe_elements: tuple, all_parsed_answers: dict, use_icl=False, use_bc=False, annotations_path=None, annotation_key=None) -> dict:
    all_parsed_answers["test_bev_frontal"] = {}
    y_pred_rgb, y_pred_ss, y_test = [], [], []
    total_inference_time = 0

    for i, test in enumerate(test_list):
        if not test.get("real_image", None):
            all_parsed_answers["test_bev_frontal"][f"test_case_{i}"] = {}

            prompt = PROMPT_TEST_BEV_FRONTAL

            if use_bc:
                possible_branches = apply_branch_classifier(test)
                prompt = prompt + f"\n**HINT**: Available options in this scenario are {possible_branches[0]} and {possible_branches[1]}."

                print(prompt)

            # ===== SS TURN =====
            images_ss = [Image.open(test["front_imgs_ss"]), Image.open(test["map_img_ss"])]
            start_time = time.time()
            raw_answer = call_lmm(pipe_elements, prompt, images=images_ss, icl_enabled=use_icl, annotations_path=annotations_path, annotation_key="annotation_frontal_bev", front_img_type="ss")
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            parsed_answer = parse_llm_response(raw_answer)
            all_parsed_answers["test_bev_frontal"][f"test_case_{i}"]["ss"] = parsed_answer
            actionAsNumber = action2number(parsed_answer["choice"])

            while actionAsNumber == -1:
                raw_answer = call_lmm(pipe_elements, prompt, images=images_ss, icl_enabled=use_icl, annotations_path=annotations_path, annotation_key="annotation_frontal_bev", front_img_type="ss")
                parsed_answer = parse_llm_response(raw_answer)
                all_parsed_answers["test_bev_frontal"][f"test_case_{i}"]["ss"] = parsed_answer
                actionAsNumber = action2number(parsed_answer["choice"])
            y_pred_ss.append(actionAsNumber)

            y_test.append(action2number(test["ground_truth"]["correct_exit"])) 

    if len(test_list) > 0: total_inference_time /= (len(test_list) * 2)
    print("Finished test_bev_frontal.")
    try:
        _save_wrong_classified(pipe_elements[0], "test_bev_frontal", test_list, y_test, {"rgb": y_pred_rgb, "ss": y_pred_ss})
    except Exception as e:
        print(f"Error saving wrong classified (test_bev_frontal): {e}")
    return _pack_results(y_test, y_pred_rgb, y_pred_ss, total_inference_time, pipe_elements[0], "test_bev_frontal")

def test_bev_coord(test_list: list, pipe_elements: tuple, all_parsed_answers: dict, use_icl=False, use_bc=False, annotations_path=None, annotation_key=None) -> dict:
    all_parsed_answers["test_bev_coord"] = {}
    y_pred_rgb, y_pred_ss, y_test = [], [], []
    total_inference_time = 0

    for i, test in enumerate(test_list):
        if not test.get("real_image", None):
            all_parsed_answers["test_bev_coord"][f"test_case_{i}"] = {}
            origin, dest = test["origin_position"], test["destination_position"]
            prompt_with_coords = PROMPT_TEST_BEV_COORD.replace(
                "image pixel coordinates.",
                f"image pixel coordinates.\n\nOrigin coordinates: ({origin['x']:.2f}, {origin['y']:.2f})\nDestination coordinates: ({dest['x']:.2f}, {dest['y']:.2f})\n"
            )

            prompt = prompt_with_coords

            if use_bc:
                possible_branches = apply_branch_classifier(test)
                prompt = prompt + f"\n**HINT**: Available options in this scenario are {possible_branches[0]} and {possible_branches[1]}."

                print(prompt)

            # ===== SS =====
            start_time = time.time()
            raw_answer = call_lmm(pipe_elements, prompt, images=[Image.open(test["map_img_ss"])], icl_enabled=use_icl, annotations_path=annotations_path, annotation_key="annotation_bev_coord", front_img_type=None)
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            parsed_answer = parse_llm_response(raw_answer)
            all_parsed_answers["test_bev_coord"][f"test_case_{i}"]["ss"] = parsed_answer
            actionAsNumber = action2number(parsed_answer["choice"])

            while actionAsNumber == -1:
                raw_answer = call_lmm(pipe_elements, prompt, images=[Image.open(test["map_img_ss"])], icl_enabled=use_icl, annotations_path=annotations_path, annotation_key="annotation_bev_coord", front_img_type=None)
                parsed_answer = parse_llm_response(raw_answer)
                all_parsed_answers["test_bev_coord"][f"test_case_{i}"]["ss"] = parsed_answer
                actionAsNumber = action2number(parsed_answer["choice"])
            y_pred_ss.append(actionAsNumber)

            y_test.append(action2number(test["ground_truth"]["correct_exit"]))

    if len(test_list) > 0: total_inference_time /= (len(test_list) * 2)
    print("Finished test_bev_coord.")
    try:
        _save_wrong_classified(pipe_elements[0], "test_bev_coord", test_list, y_test, {"rgb": y_pred_rgb, "ss": y_pred_ss})
    except Exception as e:
        print(f"Error saving wrong classified (test_bev_coord): {e}")
    return _pack_results(y_test, y_pred_rgb, y_pred_ss, total_inference_time, pipe_elements[0], "test_bev_coord")

def test_bev_frontal_coord(test_list: list, pipe_elements: tuple, all_parsed_answers: dict, use_icl=False, use_bc=False, annotations_path=None, annotation_key=None) -> dict:
    all_parsed_answers["test_bev_frontal_coord"] = {}
    y_pred_rgb, y_pred_ss, y_test = [], [], []
    total_inference_time = 0

    for i, test in enumerate(test_list):
        if not test.get("real_image", None):
            all_parsed_answers["test_bev_frontal_coord"][f"test_case_{i}"] = {}
            origin, dest = test["origin_position"], test["destination_position"]
            prompt_with_coords = PROMPT_TEST_BEV_FRONTAL_COORD.replace(
                "image pixel coordinates.",
                f"image pixel coordinates.\n\nOrigin coordinates: ({origin['x']:.2f}, {origin['y']:.2f})\nDestination coordinates: ({dest['x']:.2f}, {dest['y']:.2f})\n"
            )

            prompt = prompt_with_coords

            if use_bc:
                possible_branches = apply_branch_classifier(test)
                prompt = prompt + f"\n**HINT**: Available options in this scenario are {possible_branches[0]} and {possible_branches[1]}."

                print(prompt)

            # ===== SS =====
            images_ss = [Image.open(test["front_imgs_ss"]), Image.open(test["map_img_ss"])]
            start_time = time.time()
            raw_answer = call_lmm(pipe_elements, prompt, images=images_ss, icl_enabled=use_icl, annotations_path=annotations_path, annotation_key="annotation_bev_frontal_coord", front_img_type="ss")
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            parsed_answer = parse_llm_response(raw_answer)
            all_parsed_answers["test_bev_frontal_coord"][f"test_case_{i}"]["ss"] = parsed_answer
            actionAsNumber = action2number(parsed_answer["choice"])

            while actionAsNumber == -1:
                raw_answer = call_lmm(pipe_elements, prompt, images=images_ss, icl_enabled=use_icl, annotations_path=annotations_path, annotation_key="annotation_bev_frontal_coord", front_img_type="ss")
                parsed_answer = parse_llm_response(raw_answer)
                all_parsed_answers["test_bev_frontal_coord"][f"test_case_{i}"]["ss"] = parsed_answer
                actionAsNumber = action2number(parsed_answer["choice"])
            y_pred_ss.append(actionAsNumber)

            y_test.append(action2number(test["ground_truth"]["correct_exit"]))

    if len(test_list) > 0: total_inference_time /= (len(test_list) * 2)
    print("Finished test_bev_frontal_coord.")
    try:
        _save_wrong_classified(pipe_elements[0], "test_bev_frontal_coord", test_list, y_test, {"rgb": y_pred_rgb, "ss": y_pred_ss})
    except Exception as e:
        print(f"Error saving wrong classified (test_bev_frontal_coord): {e}")
    return _pack_results(y_test, y_pred_rgb, y_pred_ss, total_inference_time, pipe_elements[0], "test_bev_frontal_coord")

def test_bev_frontal_gm(test_list: list, pipe_elements: tuple, all_parsed_answers: dict, use_icl=False, use_bc=False, annotations_path=None) -> dict:
    all_parsed_answers["test_bev_frontal_gm"] = {}
    y_pred_gm, y_test = [], []
    total_inference_time = 0

    for i, test in enumerate(test_list):
        if not test.get("real_image", None):
            all_parsed_answers["test_bev_frontal_gm"][f"test_case_{i}"] = {}

            prompt = PROMPT_TEST_BEV_FRONTAL

            if use_bc:
                possible_branches = apply_branch_classifier(test)
                prompt = prompt + f"\n**HINT**: Available options in this scenario are {possible_branches[0]} and {possible_branches[1]}."

                print(prompt)

            images_gm = [Image.open(test["front_imgs_rgb"]), Image.open(test["map_img_ss"])]
            
            start_time = time.time()
            raw_answer = call_lmm(pipe_elements, prompt, images=images_gm, icl_enabled=use_icl, annotations_path=annotations_path, annotation_key="annotation_frontal_bev", front_img_type="rgb")
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            parsed_answer = parse_llm_response(raw_answer)
            all_parsed_answers["test_bev_frontal_gm"][f"test_case_{i}"] = parsed_answer
            actionAsNumber = action2number(parsed_answer["choice"])

            while actionAsNumber == -1:
                raw_answer = call_lmm(pipe_elements, prompt, images=images_gm, icl_enabled=use_icl, annotations_path=annotations_path, annotation_key="annotation_frontal_bev", front_img_type="rgb")
                parsed_answer = parse_llm_response(raw_answer)
                all_parsed_answers["test_bev_frontal_gm"][f"test_case_{i}"] = parsed_answer
                actionAsNumber = action2number(parsed_answer["choice"])
            y_pred_gm.append(actionAsNumber)

            y_test.append(action2number(test["ground_truth"]["correct_exit"]))

    if len(test_list) > 0: total_inference_time /= len(test_list)
    print("Finished test_bev_frontal_gm.")
    try:
        _save_wrong_classified(pipe_elements[0], "test_bev_frontal_gm", test_list, y_test, {"gm": y_pred_gm})
    except Exception as e:
        print(f"Error saving wrong classified (test_bev_frontal_gm): {e}")
    print(f"TOTAL INFERENCE TIME: {total_inference_time}")
    acc_gm, f1_gm, precision_gm, recall_gm, conf_mat_gm = show_test_results_gm(y_test, y_pred_gm, pipe_elements[0], "test_bev_frontal_gm")
    return {"y_pred_gm": y_pred_gm, "y_test": y_test, "total_inference_time": total_inference_time,
            "acc_gm": acc_gm, "f1_gm": f1_gm, "precision_gm": precision_gm, "recall_gm": recall_gm, "conf_mat_gm": conf_mat_gm}

def test_bev_frontal_gm_coord(test_list: list, pipe_elements: tuple, all_parsed_answers: dict, use_icl=False, use_bc=False, annotations_path=None) -> dict:
    all_parsed_answers["test_bev_frontal_gm_coord"] = {}
    y_pred_gm, y_test = [], []
    total_inference_time = 0

    for i, test in enumerate(test_list):
        if not test.get("real_image", None):
            all_parsed_answers["test_bev_frontal_gm_coord"][f"test_case_{i}"] = {}
            origin, dest = test["origin_position"], test["destination_position"]
            prompt_with_coords = PROMPT_TEST_BEV_FRONTAL_COORD.replace(
                "image pixel coordinates.",
                f"image pixel coordinates.\n\nOrigin coordinates: ({origin['x']:.2f}, {origin['y']:.2f})\nDestination coordinates: ({dest['x']:.2f}, {dest['y']:.2f})\n"
            )

            prompt = prompt_with_coords

            if use_bc:
                possible_branches = apply_branch_classifier(test)
                prompt = prompt + f"\n**HINT**: Available options in this scenario are {possible_branches[0]} and {possible_branches[1]}."

                print(prompt)

            images_gm = [Image.open(test["front_imgs_rgb"]), Image.open(test["map_img_ss"])]

            start_time = time.time()
            raw_answer = call_lmm(pipe_elements, prompt, images=images_gm, icl_enabled=use_icl, annotations_path=annotations_path, annotation_key="annotation_bev_frontal_coord", front_img_type="rgb")
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            parsed_answer = parse_llm_response(raw_answer)
            all_parsed_answers["test_bev_frontal_gm_coord"][f"test_case_{i}"] = parsed_answer
            actionAsNumber = action2number(parsed_answer["choice"])

            while actionAsNumber == -1:
                raw_answer = call_lmm(pipe_elements, prompt, images=images_gm, icl_enabled=use_icl, annotations_path=annotations_path, annotation_key="annotation_bev_frontal_coord", front_img_type="rgb")
                parsed_answer = parse_llm_response(raw_answer)
                all_parsed_answers["test_bev_frontal_gm_coord"][f"test_case_{i}"] = parsed_answer
                actionAsNumber = action2number(parsed_answer["choice"])
            y_pred_gm.append(actionAsNumber)

            y_test.append(action2number(test["ground_truth"]["correct_exit"]))

    if len(test_list) > 0: total_inference_time /= len(test_list)
    print("Finished test_bev_frontal_gm_coord.")
    try:
        _save_wrong_classified(pipe_elements[0], "test_bev_frontal_gm_coord", test_list, y_test, {"gm": y_pred_gm})
    except Exception as e:
        print(f"Error saving wrong classified (test_bev_frontal_gm_coord): {e}")
    acc_gm, f1_gm, precision_gm, recall_gm, conf_mat_gm = show_test_results_gm(y_test, y_pred_gm, pipe_elements[0], "test_bev_frontal_gm_coord")
    return {"y_pred_gm": y_pred_gm, "y_test": y_test, "total_inference_time": total_inference_time,
            "acc_gm": acc_gm, "f1_gm": f1_gm, "precision_gm": precision_gm, "recall_gm": recall_gm, "conf_mat_gm": conf_mat_gm}

def _pack_results(y_test, y_pred_rgb, y_pred_ss, total_inference_time, model_name, test_name):
    print(f"TOTAL INFERENCE TIME PER SAMPLE: {total_inference_time:.4f} seconds")
    acc_rgb, acc_ss, f1_rgb, f1_ss, precision_rgb, precision_ss, recall_rgb, recall_ss, cm_rgb, cm_ss = show_test_results(y_test, y_pred_rgb, y_pred_ss, model_name, test_name)
    return {
        "y_pred_rgb": y_pred_rgb, "y_pred_ss": y_pred_ss, "y_test": y_test, "total_inference_time": total_inference_time,
        "acc_rgb": acc_rgb, "acc_ss": acc_ss, "f1_rgb": f1_rgb, "f1_ss": f1_ss,
        "precision_rgb": precision_rgb, "precision_ss": precision_ss,
        "recall_rgb": recall_rgb, "recall_ss": recall_ss, "conf_mat_rgb": cm_rgb, "conf_mat_ss": cm_ss
    }

# ========================================================================================================
# MAIN
# ========================================================================================================

PROMPT_TEST_BEV, PROMPT_TEST_BEV_FRONTAL, PROMPT_TEST_BEV_COORD, PROMPT_TEST_BEV_FRONTAL_COORD = None, None, None, None
BRANCH_CLASSIFIER = None

def load_prompts(args):
    global PROMPT_TEST_BEV, PROMPT_TEST_BEV_FRONTAL, PROMPT_TEST_BEV_COORD, PROMPT_TEST_BEV_FRONTAL_COORD
    prompt_paths = {
        "PROMPT_TEST_BEV": args.test_bev,
        "PROMPT_TEST_BEV_FRONTAL": args.test_bev_frontal,
        "PROMPT_TEST_BEV_COORD": args.test_bev_coord,
        "PROMPT_TEST_BEV_FRONTAL_COORD": args.test_bev_frontal_coord,
    }
    try:
        with open(prompt_paths["PROMPT_TEST_BEV"], 'r', encoding='utf-8') as f: PROMPT_TEST_BEV = f.read()
        with open(prompt_paths["PROMPT_TEST_BEV_FRONTAL"], 'r', encoding='utf-8') as f: PROMPT_TEST_BEV_FRONTAL = f.read()
        with open(prompt_paths["PROMPT_TEST_BEV_COORD"], 'r', encoding='utf-8') as f: PROMPT_TEST_BEV_COORD = f.read()
        with open(prompt_paths["PROMPT_TEST_BEV_FRONTAL_COORD"], 'r', encoding='utf-8') as f: PROMPT_TEST_BEV_FRONTAL_COORD = f.read()
    except Exception as e:
        print(f"❌ Error loading prompts: {e}")

def save_results(model_name, all_results, all_parsed_answers):
    path = f"../carla_test/test_results/{model_name}"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/results.json", 'w', encoding='utf-8') as f: json.dump(all_results, f, ensure_ascii=False, indent=4)
    with open(f"{path}/answers.json", 'w', encoding='utf-8') as f: json.dump(all_parsed_answers, f, ensure_ascii=False, indent=4)

def main():
    cwd = os.path.basename(os.getcwd())
    print(f"Current working directory: {cwd}")
    assert cwd == "CARLA_scripts", "Working directory has to be CARLA_scripts"

    parser = argparse.ArgumentParser(description='Test models using CARLA data')
    parser.add_argument('--test-bev', default="../carla_test/prompts/PROMPT_TEST_BEV.txt")
    parser.add_argument('--test-bev-frontal', default="../carla_test/prompts/PROMPT_TEST_BEV_FRONTAL.txt")
    parser.add_argument('--test-bev-coord', default="../carla_test/prompts/PROMPT_TEST_BEV_COORD.txt")
    parser.add_argument('--test-bev-frontal-coord', default="../carla_test/prompts/PROMPT_TEST_BEV_FRONTAL_COORD.txt")
    parser.add_argument('--use-icl', action='store_true', default=False, help='Use In-Context Learning')
    parser.add_argument('--use-branch-classifier', action='store_true', default=False, help='Use a pre-trained Linear Regression to identify possible branches in an intersection')
    parser.add_argument('--annotations-icl', default="./annotations_cot.json", help='Path to annotations_cot.json for ICL')
    parser.add_argument('--do-tests', nargs='+', help='List of test to validate model')
    parser.add_argument('--model', help='Model to validate', default=None)
    parser.add_argument('--auto-test', help="Automatically select the test based on the lora model name")
    args = parser.parse_args()

    load_prompts(args)

    model_name_av = {
        0: "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        1: "google/gemma-3-12b-it",
        2: "google/gemma-3-27b-it",
        3: "OpenGVLab/InternVL3_5-38B-HF",
        4: "Qwen/Qwen3-VL-32B-Instruct",
        5: "Qwen/Qwen3-VL-32B-Thinking",
        6: "google/gemini-2.5-flash",
        8: "gemma-3-12b-it-lora-ss",
        9: "gemma-3-12b-it-lora-gm",
        10: "gemma-3-12b-it-lora-only-bev-ss",
        11: "gemma-3-12b-it-lora-gm-coord",
        12: "gemma-3-12b-it-lora-ss-verify",
        13: "gemma-3-12b-it-lora-ss-coord",
        14: "LLaVA-OneVision-1.5-8B-Instruct-lora-ss",
        15: "LLaVA-OneVision-1.5-8B-Instruct-lora-ss-coord",
        16: "LLaVA-OneVision-1.5-8B-Instruct-lora-gm",
        17: "LLaVA-OneVision-1.5-8B-Instruct-lora-gm-coord",
        18: "LLaVA-OneVision-1.5-8B-Instruct-lora-only-bev-ss",
        19: "LLaVA-OneVision-1.5-8B-Instruct-lora-bev-ss-coord",
        20: "Qwen3-VL-32B-Instruct-lora-only-bev-ss",
        21: "Qwen3-VL-32B-Instruct-lora-ss",
        22: "Qwen3-VL-32B-Instruct-lora-bev-ss-coord",
        23: "Qwen3-VL-32B-Instruct-lora-gm",
        24: "Qwen3-VL-32B-Instruct-lora-gm-coord",
        25: "Qwen3-VL-32B-Instruct-lora-ss-coord",
        26: "CIL_customed_ss",
        27: "CIL_customed_gm",
        33: "Qwen/Qwen3-VL-235B-A22B-Instruct",
        34: "LLaVA-OneVision-1.5-8B-Instruct-lora-gm-VISION",
        35: "LLaVA-OneVision-1.5-8B-Instruct-lora-gm-VISION-R64-A128",
        36: "LLaVA-OneVision-1.5-8B-Instruct-lora-only-bev-ss-VISION",
        37: "LLaVA-OneVision-1.5-8B-Instruct-lora-ss-VISION",
        38: "LLaVA-OneVision-1.5-8B-Instruct-lora-bev-ss-coord-VISION",
        39: "LLaVA-OneVision-1.5-8B-Instruct-lora-ss-coord-VISION",
        40: "LLaVA-OneVision-1.5-8B-Instruct-lora-gm-coord-VISION",
        41: "gemma-3-12b-it-lora-bev-ss-coord",
        42: "Qwen3-VL-32B-Instruct-lora-only-bev-ss-VISION",
        43: "Qwen3-VL-32B-Instruct-lora-bev-ss-coord-VISION",
        44: "Qwen3-VL-32B-Instruct-lora-gm-VISION",
        45: "Qwen3-VL-32B-Instruct-lora-gm-coord-VISION",
        46: "Qwen3-VL-32B-Instruct-lora-ss-VISION",
        47: "Qwen3-VL-32B-Instruct-lora-ss-coord-VISION",
        50: "gemma-3-12b-it-lora-only-bev-ss-VISION",
        51: "gemma-3-12b-it-lora-bev-ss-coord-VISION",
        52: "gemma-3-12b-it-lora-gm-coord-VISION",
        53: "gemma-3-12b-it-lora-ss-coord-VISION",
        54: "gemma-3-12b-it-lora-gm-VISION",
        55: "gemma-3-12b-it-lora-ss-VISION",
        70: "LLaVA-OneVision-1.5-8B-Instruct-lora-only-bev-ss-LANGUAGE",
        71: "LLaVA-OneVision-1.5-8B-Instruct-lora-bev-ss-coord-LANGUAGE",
        72: "LLaVA-OneVision-1.5-8B-Instruct-lora-ss-LANGUAGE",
        73: "LLaVA-OneVision-1.5-8B-Instruct-lora-ss-coord-LANGUAGE",
        74: "LLaVA-OneVision-1.5-8B-Instruct-lora-gm-LANGUAGE",
        75: "LLaVA-OneVision-1.5-8B-Instruct-lora-gm-coord-LANGUAGE",
        76: "Qwen3-VL-32B-Instruct-lora-only-bev-ss-LANGUAGE",
        77: "Qwen3-VL-32B-Instruct-lora-bev-ss-coord-LANGUAGE",
        78: "Qwen3-VL-32B-Instruct-lora-ss-LANGUAGE",
        79: "Qwen3-VL-32B-Instruct-lora-ss-coord-LANGUAGE",
        80: "Qwen3-VL-32B-Instruct-lora-gm-LANGUAGE",
        81: "Qwen3-VL-32B-Instruct-lora-gm-coord-LANGUAGE",
        86: "gemma-3-12b-it-lora-only-bev-ss-LANGUAGE",
        87: "gemma-3-12b-it-lora-bev-ss-coord-LANGUAGE",
        88: "gemma-3-12b-it-lora-ss-LANGUAGE",
        89: "gemma-3-12b-it-lora-ss-coord-LANGUAGE",
        90: "gemma-3-12b-it-lora-gm-LANGUAGE",
        91: "gemma-3-12b-it-lora-gm-coord-LANGUAGE",
        100: "LLaVA-OneVision-1.5-8B-Instruct-lora-only-bev-ss-CNN"
    }

    model_name = model_name_av[27] 

    if args.model!=None:
        if args.model not in list(model_name_av.values()): raise ValueError("Invalid model")
        model_name = args.model


    for m in model_name_av.values():
        os.makedirs(os.path.join("..", "carla_test", "test_results", m), exist_ok=True)

    pipe_elements = load_lmm(model_name)

    with open("./data_updated.json", "r") as f:
        dataset = json.load(f)

    train, validation = split_dataset(dataset)
    validation = get_balanced_subset(validation, num_samples=999, random_seed=7)

    print(f"Testing on {len(validation)} samples.")
    
    all_parsed_answers = {}
    all_results = {}

    if args.use_branch_classifier:
        global BRANCH_CLASSIFIER
        print("Loading branch classifier...")
        with open("./dataset_branches_classifier/branch_classifier.pkl", "rb") as f:
            BRANCH_CLASSIFIER = pickle.load(f)
        print(BRANCH_CLASSIFIER)

    # --- UNCOMMENT SECTIONS TO RUN TESTS ---
    args.auto_test=True
    if args.auto_test and model_name.find("lora-only-bev-ss")!=-1 or "test_bev" in args.do_tests:
        print("Running test BEV...")
        all_results['test_bev'] = test_bev(validation, pipe_elements, all_parsed_answers, 
                                        use_icl=args.use_icl, use_bc=args.use_branch_classifier, annotations_path=args.annotations_icl)
        save_results(model_name if not args.use_icl else model_name+"_icl", all_results, all_parsed_answers)    

    if args.auto_test and model_name.find("lora-ss")!=-1 and model_name.find("coord")==-1 or "test_bev_frontal" in args.do_tests:
        print("Running test BEV + Frontal...")
        all_results['test_bev_frontal'] = test_bev_frontal(validation, pipe_elements, all_parsed_answers, 
                                                        use_icl=args.use_icl, use_bc=args.use_branch_classifier, annotations_path=args.annotations_icl)
        save_results(model_name if not args.use_icl else model_name+"_icl", all_results, all_parsed_answers)

    if args.auto_test and model_name.find("lora-bev-ss-coord")!=-1 or "test_bev_coord" in args.do_tests:
        print("Running test BEV + COORD...")
        all_results['test_bev_coord'] = test_bev_coord(validation, pipe_elements, all_parsed_answers, 
                                                    use_icl=args.use_icl, use_bc=args.use_branch_classifier, annotations_path=args.annotations_icl)
        save_results(model_name if not args.use_icl else model_name+"_icl", all_results, all_parsed_answers)

    if args.auto_test and model_name.find("lora-ss-coord")!=-1 or "test_bev_frontal_coord" in args.do_tests:
        print("Running test BEV + FRONTAL + COORD...")
        all_results['test_bev_frontal_coord'] = test_bev_frontal_coord(validation, pipe_elements, all_parsed_answers,
                                                                    use_icl=args.use_icl, use_bc=args.use_branch_classifier, annotations_path=args.annotations_icl)
        save_results(model_name if not args.use_icl else model_name+"_icl", all_results, all_parsed_answers)

    if args.auto_test and model_name.find("lora-gm")!=-1 and model_name.find("coord")==-1 or "test_bev_frontal_gm" in args.do_tests:
        print("Running test BEV + FRONTAL GM...")
        all_results['test_bev_frontal_gm'] = test_bev_frontal_gm(validation, pipe_elements, all_parsed_answers,
                                                                use_icl=args.use_icl, use_bc=args.use_branch_classifier, annotations_path=args.annotations_icl)
        save_results(model_name if not args.use_icl else model_name+"_icl", all_results, all_parsed_answers)

    if args.auto_test and model_name.find("lora-gm-coord")!=-1 or "test_bev_frontal_gm_coord" in args.do_tests:
        print("Running test BEV + FRONTAL GM + COORD...")
        all_results['test_bev_frontal_gm_coord'] = test_bev_frontal_gm_coord(validation, pipe_elements, all_parsed_answers,
                                                                            use_icl=args.use_icl, use_bc=args.use_branch_classifier, annotations_path=args.annotations_icl)
        save_results(model_name if not args.use_icl else model_name+"_icl", all_results, all_parsed_answers)

if __name__=="__main__":
    main()
