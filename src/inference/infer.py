import segmentation_models_pytorch as smp
import torch

def load_model(checkpoint_path, device='cpu'):
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

if __name__ == "__main__":
    model = load_model('models/best_model_v2_safe.pth')
    print("Model loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")