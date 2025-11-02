import torch
import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Resize
from models.model_single import ModelEmb
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from train import get_input_dict, norm_batch, get_dice_ji


def preprocess_image(image_path, resize_dim):
    """
    Preprocess the input image: resize and convert to tensor.
    """
    img = Image.open(image_path).convert("RGB")
    transform = Resize(resize_dim)
    img = transform(img)
    img_tensor = ToTensor()(img)
    return img_tensor.unsqueeze(0)  # Add batch dimension

def load_model(model_path, sam_checkpoint, sam_model_type, device, args):
    """
    Load the trained autosam model and SAM backbone.
    """
    # Load the trained hardnet model
    model = ModelEmb(args=args).to(device)  # Provide args here
    model_state = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state.state_dict())
    model.eval()

    # Load the SAM model
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()

    return model, sam



def infer_single_image(image_path, model, sam, resize_dim, device):
    """
    Perform inference on a single image to generate a mask.
    """
    # Preprocess the image
    img_tensor = preprocess_image(image_path, resize_dim).to(device)

    # Generate dense embeddings using the trained model
    dense_embeddings = model(img_tensor)

    # Prepare input for SAM
    original_sz = img_tensor.shape[2:]
    batched_input = [
        {
            'image': img_tensor.squeeze(0),
            'original_size': original_sz,
            'image_size': original_sz,
            'point_coords': None,
            'point_labels': None,
        }
    ]

    # Generate masks using SAM
    with torch.no_grad():
        input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)
        low_res_masks, _ = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings_none,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

    # Process masks
    masks = norm_batch(low_res_masks)
    masks[masks > 0.5] = 1
    masks[masks <= 0.5] = 0

    return masks.squeeze(0).cpu().numpy()

def main():
    # Set paths and parameters
    model_path = "results/gpu1/net_best.pth"  # Adjust to the actual path
    sam_checkpoint = "cp/sam_vit_h.pth"
    sam_model_type = "vit_h"
    image_path = "dataset/polyp/TrainDataset/images/577.png"  # Path to the test image
    resize_dim = (512, 512)  # Image resize dimensions

    # Example args (adjust based on your model requirements)
    args = {
        'task': 'glas',
        'Idim': 512,
        'depth_wise': False,
        'order': 85,
    }

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    model, sam = load_model(model_path, sam_checkpoint, sam_model_type, device, args)
    # Perform inference
    mask = infer_single_image(image_path, model, sam, resize_dim, device)

    # Process mask for saving
    mask = mask.squeeze()  # Remove batch and channel dimensions
    if mask.ndim == 3 and mask.shape[0] == 1:  # Ensure no extra dimension
        mask = mask[0]
    mask = (mask * 255).astype(np.uint8)  # Scale values and convert to uint8

    # Save the mask
    output_mask_path = "output/mask.png"
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    Image.fromarray(mask).save(output_mask_path)

    print(f"Mask saved to {output_mask_path}")


if __name__ == "__main__":
    main()
