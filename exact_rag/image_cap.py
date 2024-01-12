import torch
from PIL import Image
from lavis.models import load_model_and_preprocess


def image_captioning(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads BLIP caption base model, with finetuned checkpoints
    # this also loads the associated image processors
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    raw_image = Image.open(file_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # generate caption
    text = model.generate({"image": image})
    return text
