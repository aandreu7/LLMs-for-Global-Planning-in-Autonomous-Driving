import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import importlib
import sys
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
import numpy as np
import random

# Set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Add project root to path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
sys.path.append(root_path)

# Mock tensorflow and other missing dependencies
from unittest.mock import MagicMock
mock_tf = MagicMock()
mock_tf.__spec__ = MagicMock()
sys.modules["tensorflow"] = mock_tf
sys.modules["ttach"] = MagicMock()

from network.models.building_blocks import FC
from network.models.building_blocks.PositionalEncoding import PositionalEncoding
from network.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoder
from network.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoderLayer

"""
IMPORTANT:
    Pytorch uses B,C,H,W notation with images.
"""

MODEL_CONFIGURATION = {
    "backbone": {
        "IMAGENET_PRE_TRAINED": True,
    },
    "encoder_embedding": {
        "perception": {
            "res": {
                "name": "resnet34",
                "layer_id": 4
            }
        }
    },
    "TxEncoder": {
        "d_model": 512,
        "n_head": 4,
        "num_layers": 4,
        "norm_first": True,
        "learnable_pe": True
    },
    "command": {
        "fc": {
            "neurons": [512],
            "dropouts": [0.0]
        }
    },
    "speed": {
        "fc": {
            "neurons": [512],
            "dropouts": [0.0]
        }
    },
    "action_output": {
        "fc": {
            "neurons": [512, 256],
            "dropouts": [0.0, 0.0]
        }
    }
}


class CIL_multiview(nn.Module):
    def __init__(self, params):
        super(CIL_multiview, self).__init__()
        self.params = params

        resnet_module = importlib.import_module('network.models.building_blocks.resnet_FM')
        resnet_module = getattr(resnet_module, params['encoder_embedding']['perception']['res']['name'])
        self.encoder_embedding_perception = resnet_module(pretrained=params["backbone"]["IMAGENET_PRE_TRAINED"],
                                                          layer_id = params['encoder_embedding']['perception']['res'][ 'layer_id'])

        # layer_id denotes the last layer of the ResNet backbone that we want to use, as using the complete backbone would be unnecessary

        # Calculate output shapes for both images
        # Image 1: 900x300 (WxH) -> (3, 300, 900)
        _, self.res_out_dim, self.res_out_h1, self.res_out_w1 = self.encoder_embedding_perception.get_backbone_output_shape([1, 3, 300, 900])[params['encoder_embedding']['perception']['res'][ 'layer_id']]
        # Image 2: 300x300 (WxH) -> (3, 300, 300)
        _, _, self.res_out_h2, self.res_out_w2 = self.encoder_embedding_perception.get_backbone_output_shape([1, 3, 300, 300])[params['encoder_embedding']['perception']['res'][ 'layer_id']]

        # res_out_dim --> Number of filters = Number of output channels

        # Total tokens = tokens from img1 + tokens from img2
        total_tokens = (self.res_out_h1 * self.res_out_w1) + (self.res_out_h2 * self.res_out_w2)

        # Each token is a C dimensional vector (C = n_dim, C would be the embedding vector)

        if params['TxEncoder']['learnable_pe']:
            self.positional_encoding = nn.Parameter(torch.zeros(1, total_tokens, params['TxEncoder']['d_model']))
        else:
            self.positional_encoding = PositionalEncoding(d_model=params['TxEncoder']['d_model'], dropout=0.0, max_len=total_tokens)

        join_dim = params['TxEncoder']['d_model']

        # self.command = nn.Linear(g_conf.DATA_COMMAND_CLASS_NUM, params['TxEncoder']['d_model'])
        # self.speed = nn.Linear(1, params['TxEncoder']['d_model'])

        tx_encoder_layer = TransformerEncoderLayer(d_model=params['TxEncoder']['d_model'],
                                                   nhead=params['TxEncoder']['n_head'],
                                                   norm_first=params['TxEncoder']['norm_first'], batch_first=True)
        self.tx_encoder = TransformerEncoder(tx_encoder_layer, num_layers=params['TxEncoder']['num_layers'],
                                             norm=nn.LayerNorm(params['TxEncoder']['d_model']))

        # Output: 3 classes (Straight, Right, Left)
        self.action_output = FC(params={'neurons': [join_dim] +
                                            params['action_output']['fc']['neurons'] +
                                            [3], # 3 classes
                                 'dropouts': params['action_output']['fc']['dropouts'] + [0.0],
                                 'end_layer': True})

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

        self.train()

        
        # Transforms used to convert PIL images into tensors (only used when inference from PIL images)
        self.transform1 = T.Compose([
            T.Resize((300, 900)),
            T.ToTensor()
        ])

        self.transform2 = T.Compose([
            T.Resize((300, 300)),
            T.ToTensor()
        ])

    def inference_from_pil(self, img1, img2):
        # 1. Cargar y transformar imágenes
        img1 = self.transform1(img1)
        img2 = self.transform2(img2)
        
        # 2. Agregar dimensión de lote (B=1)
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        
        # 3. Mover a GPU/CPU
        device = next(self.parameters()).device
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        # 4. Forward
        output = self(img1, img2)

        # 5. Procesar la salida
        output = output.squeeze()    # shape: [3]
        
        # Obtener la clase predicha (índice 0, 1, o 2)
        predicted_class = torch.argmax(output).item()
        
        predicted_class = "Straight" if predicted_class == 0 else "Right" if predicted_class == 1 else "Left" if predicted_class == 2 else -1

        return predicted_class


    def forward(self, img1, img2):
        # img1: [B, 3, 300, 900]
        # img2: [B, 3, 300, 300]
        B = img1.shape[0]

        # Process Image 1
        e_p1, _ = self.encoder_embedding_perception(img1) # [B, dim, h1, w1]
        encoded_obs1 = e_p1.view(B, self.res_out_dim, self.res_out_h1 * self.res_out_w1)
        encoded_obs1 = encoded_obs1.transpose(1, 2) # [B, h1*w1, dim]

        # Process Image 2
        e_p2, _ = self.encoder_embedding_perception(img2) # [B, dim, h2, w2]
        encoded_obs2 = e_p2.view(B, self.res_out_dim, self.res_out_h2 * self.res_out_w2)
        encoded_obs2 = encoded_obs2.transpose(1, 2) # [B, h2*w2, dim]

        # Concatenate tokens
        encoded_obs = torch.cat([encoded_obs1, encoded_obs2], dim=1) # [B, total_tokens, dim]

        # e_d = self.command(d).unsqueeze(1)     # [B, 1, 512]
        # e_s = self.speed(s).unsqueeze(1)       # [B, 1, 512]

        # encoded_obs = encoded_obs + e_d + e_s
        # No command/speed addition

        if self.params['TxEncoder']['learnable_pe']:
            # positional encoding
            pe = encoded_obs + self.positional_encoding    # [B, total_tokens, 512]
        else:
            pe = self.positional_encoding(encoded_obs)

        # Transformer encoder multi-head self-attention layers
        in_memory, _ = self.tx_encoder(pe)  # [B, total_tokens, 512]

        in_memory = torch.mean(in_memory, dim=1)  # [B, 512]

        action_output = self.action_output(in_memory).unsqueeze(1)  # (B, 512) -> (B, 1, 3)

        return action_output         # (B, 1, 3)

    def foward_eval(self, img1, img2):
        # img1: [B, 3, 300, 900]
        # img2: [B, 3, 300, 300]
        B = img1.shape[0]

        # Process Image 1
        e_p1, resnet_inter1 = self.encoder_embedding_perception(img1)
        encoded_obs1 = e_p1.view(B, self.res_out_dim, self.res_out_h1 * self.res_out_w1)
        encoded_obs1 = encoded_obs1.transpose(1, 2)

        # Process Image 2
        e_p2, resnet_inter2 = self.encoder_embedding_perception(img2)
        encoded_obs2 = e_p2.view(B, self.res_out_dim, self.res_out_h2 * self.res_out_w2)
        encoded_obs2 = encoded_obs2.transpose(1, 2)

        encoded_obs = torch.cat([encoded_obs1, encoded_obs2], dim=1)

        # e_d = self.command(d).unsqueeze(1)  # [B, 1, 512]
        # e_s = self.speed(s).unsqueeze(1)  # [B, 1, 512]
        # encoded_obs = encoded_obs + e_d + e_s

        if self.params['TxEncoder']['learnable_pe']:
            # positional encoding
            pe = encoded_obs + self.positional_encoding
        else:
            pe = self.positional_encoding(encoded_obs)

        # Transformer encoder multi-head self-attention layers
        in_memory, attn_weights = self.tx_encoder(pe)
        in_memory = torch.mean(in_memory, dim=1)

        action_output = self.action_output(in_memory).unsqueeze(1)

        return action_output, (resnet_inter1, resnet_inter2), attn_weights

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



def fit(model, train_loader: DataLoader, num_epochs: int = 100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"USING: {device}")
    
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for img1, img2, labels in train_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward
            outputs = model(img1, img2) # [B, 1, 3]
            outputs = outputs.squeeze(1) # [B, 3]

            loss = criterion(outputs, labels)

            # Backward
            loss.backward()

            optimizer.step() # Params. update

            total_loss += loss.item()

        scheduler.step() # Learning rate update (decreases it)

        # Logs
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}  Avg Loss: {avg_loss:.4f}")
        with open("/datafast/105-1/Datasets/INTERNS/aplanaj/CIL_customed_training_logs.txt", "a") as f:
            f.write(f"{epoch}, {loss.item()}, {optimizer.param_groups[0]['lr']}\n")


def load_gross_data():
    data_path = "data.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    samples = [x for x in data if x.get("clean_bev_ss_image") and x["clean_bev_ss_image"].split('/')[-1].split('_')[0]=="Town01"]

    SEGMENTED_FRONT = True
    front_img_json_name = "front_imgs_rgb" if SEGMENTED_FRONT else "front_imgs_ss"
    
    formatted_samples = [x[front_img_json_name], [x["map_img_ss"], x["ground_truth"]] for x in samples]

    return formatted_samples


class FrontBevTorchDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

        # FRONT IMAGE
        self.transform1 = T.Compose([
            T.Resize((300, 900)),
            T.ToTensor()
        ])

        # BEV IMAGE
        self.transform2 = T.Compose([
            T.Resize((300, 300)),
            T.ToTensor()
        ])

        # Labels (Straight/Right/Left)
        self.transform3 = lambda x: 0 if x == "Straight" else 1 if x == "Right" else 2 if x == "Left" else -1

    def __getitem__(self, idx):
        path1, path2, label = self.samples[idx]

        # FRONT IMAGE
        img1 = self.transform1(Image.open(path1).convert("RGB"))

        # BEV IMAGE
        img2 = self.transform2(Image.open(path2).convert("RGB"))

        # Labels (Straight/Right/Left)
        label = self.transform3(label)

        #return img1, img2, label
        return img1, img2, label

    def __len__(self):
        return len(self.samples)



def main():

    train_data_gross = load_gross_data()
    train_data_loader = FrontBevTorchDataset(train_data_gross)

    BATCH_SIZE = 32 # Batch/GPU = 8 (if using GPUs 0-3)
    train_data_net = DataLoader(
        train_data_loader, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = CIL_multiview(MODEL_CONFIGURATION)

    if torch.cuda.device_count() > 1:
        print(f"GPU Parallelization available. Activating DataParallel in {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    fit(model, train_data_net)

    # Save the complete model (architecture + parameters)
    # Alternative: only save the parameters using model.state_dict()
    if isinstance(model, nn.DataParallel):
        # Unwraps the model if it was wrapped into a DataParallel class
        model_to_save = model.module
    else:
        model_to_save = model
    save_path = '/datafast/105-1/Datasets/INTERNS/aplanaj/CIL_customed_gm.pth'
    torch.save(model_to_save, save_path)
    print(f"Trained model saved to {save_path}")

if __name__ == "__main__":
    cwd = os.path.basename(os.getcwd())
    print(f"Current working directory: {cwd}")
    assert cwd == "CARLA_scripts", "Working directory has to be CARLA_scripts"

    main()