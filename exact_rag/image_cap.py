import torch
from transformers import pipeline
from transformers import Pipeline


def image_captioner(image_model: str) -> Pipeline:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    captioner = pipeline("image-to-text", model=image_model, device=device)

    return captioner
