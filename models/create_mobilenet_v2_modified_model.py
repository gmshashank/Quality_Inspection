import torch
# from torchvision.models import mobilenet

IN_MODEL_FILENAME = "mobilenet_v2_cpu_traced.pt"
OUT_MODEL_FILENAME = "mobilenet_v2_modified.pt"


def create_model(out_dir: str = "./"):
    """
    Trace pre trained model into new model  
    """
    model=torch.load(f"./{IN_MODEL_FILENAME}",map_location=torch.device("cpu"))
    model=model.to("cpu")
    model.eval()
    traced_model=torch.jit.trace(model,torch.randn(1,3,224,224))
    traced_model.save(out_dir+OUT_MODEL_FILENAME)

if __name__ == "__main__":
    create_model()
    print(f"{OUT_MODEL_FILENAME} model file is created successfully.")
