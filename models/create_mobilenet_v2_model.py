import torch
from torchvision.models import mobilenet

MODEL_FILENAME = "mobilenet_v2.pt"


def create_model(out_dir: str = "./"):
    """
    Trace pre trained model into new model  
    """
    model = mobilenet.mobilenet_v2(pretrained=True)
    model.eval()
    traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
    traced_model.save(out_dir + MODEL_FILENAME)


if __name__ == "__main__":
    create_model()
    print(f"{MODEL_FILENAME} model file is created successfully.")
