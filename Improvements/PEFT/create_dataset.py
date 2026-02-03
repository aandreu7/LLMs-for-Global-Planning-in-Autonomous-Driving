import os
import json
import argparse

from datasets import Dataset
from PIL import Image
import numpy as np


def gather_data(data_json_path: str, training_map: str):
    """
    Gather data from JSON file and filter by training map.
    
    Args:
        data_json_path: Path to the JSON file containing the data
        training_map: CARLA map name to filter (e.g., "Town01")
    """
    data = []

    with open(data_json_path, "r") as f:
        data_structure = json.load(f)

    for sample_structure in data_structure:

        if not sample_structure.get("checked") or not sample_structure.get("clean_bev_ss_image"):
            continue
        town_name = sample_structure["clean_bev_ss_image"].split('/')[-1].split('_')[0]
        if town_name != training_map:
            continue

        sample = {
            #"front_path": sample_structure["front_imgs_ss"],
            "map_path": sample_structure["map_img_ss"],
            "origin_coords": sample_structure["origin_position"],
            "destination_coords": sample_structure["destination_position"],
            "label": sample_structure["ground_truth"]["correct_exit"]
        }

        data.append(sample)

    hf_dataset = Dataset.from_list(data)
    return hf_dataset



def preprocess(example, prompt_template: str):
    """
    Preprocess a single example by loading images and creating messages.
    
    Args:
        example: A single data example
        prompt_template: The prompt template to use
    """
    # 1. Cargar imágenes (PIL)
    try:
        #front_img = Image.open(example["front_path"]).convert("RGB")
        map_img = Image.open(example["map_path"]).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # 2. Construir mensajes (SOLO PLACEHOLDERS para las imágenes)
    prompt_with_coords = prompt_template.replace(
        "image pixel coordinates.",
        f"image pixel coordinates."
        f"\n\nOrigin coordinates: ({example['origin_coords']['x']:.2f}, {example['origin_coords']['y']:.2f})"
        f"\nDestination coordinates: ({example['destination_coords']['x']:.2f}, {example['destination_coords']['y']:.2f})\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                #{\"type\": \"image\"},  # Placeholder 1
                {"type": "image"},  # Placeholder 2
                {"type": "text", "text": prompt_with_coords}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{example['label']}"}
            ]
        }
    ]

    # 3. Crear la lista separada de imágenes
    # El orden debe coincidir con el orden de los placeholders arriba.
    #images_list = [front_img, map_img]
    images_list = [map_img]

    # Devolvemos AMBAS columnas
    return {
        "messages": messages, 
        "images": images_list
    }



def main():
    parser = argparse.ArgumentParser(description="Prepares dataset for training.")
    parser.add_argument(
        "--data-json-path", 
        help="Path to the JSON file containing the data", 
        type=str, 
        default="./data_updated.json"
    )
    parser.add_argument(
        "--training-map", 
        help="CARLA map used for training", 
        type=str, 
        default="Town01"
    )
    parser.add_argument(
        "--output-path",
        help="Path where the processed dataset will be saved",
        type=str,
        default="/datafast/105-1/Datasets/INTERNS/aplanaj/hf_dataset_bev_ss_coord"
    )
    parser.add_argument(
        "--prompt-template",
        help="Path to a text file containing the prompt template (optional)",
        type=str,
        default=None
    )
    
    args = parser.parse_args()

    # Load prompt template
    if args.prompt_template and os.path.exists(args.prompt_template):
        with open(args.prompt_template, 'r') as f:
            prompt_template = f.read()
    else:
        # Default prompt template
        prompt_template = """I am attaching an image showing the birds-eye-view of my surroundings.  

On the birds-eye-view, my position is indicated by a red arrow showing the direction my car is facing.  
My destination is marked with a blue square.

Additionally, you will receive numeric coordinates for the origin (current position of my car) and the destination, given as (x, y) in image pixel coordinates.

Your task is to determine the next action that I have to take with my car to reach the blue destination square, minimizing the driving distance.  
This is a classification task with three possible outputs: Right, Left, Straight.

Please respond with only one unique word (Straight/Right/Left).
"""
    
    print(f"Loading data from: {args.data_json_path}")
    print(f"Filtering by map: {args.training_map}")
    
    hf_dataset = gather_data(args.data_json_path, args.training_map)
    
    print(f"Preprocessing {len(hf_dataset)} samples...")
    hf_dataset = hf_dataset.map(
        lambda example: preprocess(example, prompt_template=prompt_template),
        remove_columns=hf_dataset.column_names  # Remover columnas originales
    )

    print(f"Saving dataset to: {args.output_path}")
    hf_dataset.save_to_disk(args.output_path)

    print(f"✅ Dataset saved successfully with {len(hf_dataset)} samples")


if __name__ == "__main__":
    main()
