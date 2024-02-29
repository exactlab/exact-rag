import torch
from transformers import BlipProcessor
from transformers import BlipForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def captioner(raw_image, model_name) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    text = "a picture of "
    inputs = processor(raw_image, text, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
