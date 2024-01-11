import toml
import whisper
import aiofiles

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from fastapi import HTTPException
from exact_rag.dataemb import DataEmbedding

# Query type
class Query(BaseModel):
    text: str

# general settings
settings = toml.load("settings.toml")
de = DataEmbedding(settings)

# settign that will be general settings
model = whisper.load_model("base")

out_file_path = "/tmp/"

app = FastAPI()

@app.post("/upload_text/")
async def upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".txt"):
        
        async with aiofiles.open(out_file_path + file.filename, 'wb') as out_file:
            while content := await file.read(1024):  # async read chunk
                await out_file.write(content)  
                   
        de.load(content)
        
        return {"filename": file.filename, 
                "contents": content.decode("utf-8"),
                "file_type": "txt"}
    
    elif file.filename.endswith(".json"):
        contents = await file.read()
        return {"filename": file.filename,
                "contents": contents.decode("utf-8"),
                "file_type": "json"}
    
    else:
        raise HTTPException(status_code=400, detail="Only .txt and .json files are allowed")

@app.post("/upload_audio/")
async def upload_audio(audio: UploadFile = File(...)):
    if audio.filename.endswith(".mp3"):
        
        async with aiofiles.open(out_file_path + audio.filename, 'wb') as out_file:
            while content := await audio.read(1024):
                await out_file.write(content) 
                 
        result = model.transcribe(out_file_path + audio.filename)
        de.load(result["text"])
        
        return {"filename": audio.filename, 
        "contents": result["text"],
        "file_type": "mp3"}
        
    else:
        raise HTTPException(status_code=400, detail="Only .mp3 files are allowed")
     

@app.post("/upload_image/")
async def upload_image(image: UploadFile = File(...)):
    if image.filename.endswith((".png", ".jpeg")):
        async with aiofiles.open(out_file_path + image.filename, 'wb') as out_file:
            while content := await image.read(1024):
                await out_file.write(content)
        
        print(out_file_path + image.filename)
        
        import torch
        from PIL import Image
        from lavis.models import load_model_and_preprocess
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # loads BLIP caption base model, with finetuned checkpoints
        # this also loads the associated image processors
        model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", 
                                                             is_eval=True, device=device)
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        raw_image = Image.open(out_file_path + image.filename).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        # generate caption
        text = model.generate({"image": image})
        de.load(text[0]) # it is a list with a single element
        print(text)


@app.post("/query/")
async def send_query(query: Query):
    ans = de.chat(query.text)
    return {"Answer": ans}