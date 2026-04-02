import torch
import yaml
import argparse
from collections import OrderedDict

# Assuming the model definitions are in the 'models' directory
from models.lightningdit import LightningDiT_models

def inspect_weights(config_path, ckpt_path, layer_names):
    """
    Loads a model and its checkpoint, then prints the specified layer weights.

    Args:
        config_path (str): Path to the model config file.
        ckpt_path (str): Path to the model checkpoint file (.pt).
        layer_names (list): A list of layer names (strings) to inspect.
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # Create model
    model = LightningDiT_models[model_config['model_type']](
        input_size=config['data']['image_size'] // 16,
        num_classes=1000,
    )
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "ema" in checkpoint:
        print("Checkpoint contains 'ema', using ema state_dict.")
        state_dict = checkpoint["ema"]
    elif "model" in checkpoint:
        print("Checkpoint contains 'model', using model state_dict.")
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Clean state_dict keys if necessary (e.g., remove 'module.' prefix)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print(f"Successfully loaded checkpoint from {ckpt_path}")

    # Get the model's state_dict
    model_state_dict = model.state_dict()

    # Print the requested layer weights and shapes
    for layer_name in layer_names:
        if layer_name in model_state_dict:
            weight = model_state_dict[layer_name]
            print(f"\n--- Layer: {layer_name} ---")
            print(f"Shape: {weight.shape}")
            print("Value:")
            print(weight)
        else:
            print(f"\n--- Layer: {layer_name} ---")
            print("Error: Layer not found in model state_dict.")
            # Optional: print available keys if layer is not found
            # print("\nAvailable keys:", list(model_state_dict.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect specific weights of a LightningDiT model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model config file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument(
        "--layers", 
        nargs='+', 
        required=True, 
        help="A space-separated list of layer names to inspect."
    )
    
    args = parser.parse_args()

    inspect_weights(args.config, args.ckpt, args.layers) 