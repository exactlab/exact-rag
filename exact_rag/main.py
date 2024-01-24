import toml
import aiofiles
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from fastapi import HTTPException

from exact_rag.dataemb import DataEmbedding
from exact_rag.config import Embeddings
from exact_rag.config import Databases
from exact_rag.schemas import Query
from exact_rag.schemas import GenericResponse
from exact_rag.schemas import Answer


# general settings
settings = toml.load("settings.toml")
output_dir = settings["server"]["output_dir"]
settings_e = settings["embedding"]
settings_d = settings["database"]
embedding = Embeddings(**settings_e)
database = Databases(**settings_d)
de = DataEmbedding(embedding, database)


ml_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    whisper_model = settings["audio"].get("whisper_model")
    image_model = settings["image"].get("image_model")

    if whisper_model is not None:
        try:
            from whisper import load_model as whisper_load_model  # noqa: F401

            ml_models["whisper_model"] = whisper_load_model(whisper_model)

        except ModuleNotFoundError:
            raise BaseException("You had to install `audio` extra")

    if image_model is not None:
        try:
            import transformers  # noqa: F401
            import PIL  # noqa: F401
            from exact_rag.image_cap import image_captioner

            ml_models["image_model"] = image_captioner(image_model)

        except ModuleNotFoundError:
            raise BaseException("You had to install `image` extra")

    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/upload_text/", response_model=GenericResponse, status_code=201)
async def upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".txt"):
        async with aiofiles.open(output_dir + file.filename, "wb") as out_file:
            while content := await file.read(1024):  # async read chunk
                await out_file.write(content)

        de.load(content)

        return dict(filename=file.filename)

    elif file.filename.endswith(".json"):
        # contents = await file.read()
        return dict(filename=file.filename)

    else:
        raise HTTPException(
            status_code=400, detail="Only .txt and .json files are allowed"
        )


@app.post("/upload_audio/", response_model=GenericResponse, status_code=201)
async def upload_audio(audio: UploadFile = File(...)):
    try:
        from exact_rag.audio_cap import audio_caption

        if audio.filename.endswith(".mp3"):
            async with aiofiles.open(output_dir + audio.filename, "wb") as out_file:
                while content := await audio.read(1024):
                    await out_file.write(content)

            if ml_models["whisper_model"] is None:
                raise HTTPException(
                    status_code=400, detail="A whisper model is required, es. 'base'"
                )

            result = audio_caption(output_dir + audio.filename)
            de.load(result["text"])

            return dict(filename=audio.filename)
        else:
            raise HTTPException(status_code=400, detail="Only .mp3 files are allowed")
    except ModuleNotFoundError:
        raise BaseException("To use this endpoint you should install `audio` extra")


@app.post("/upload_image/", response_model=GenericResponse, status_code=201)
async def upload_image(image: UploadFile = File(...)):
    if image.filename.endswith((".png", ".jpeg")):
        async with aiofiles.open(output_dir + image.filename, "wb") as out_file:
            while content := await image.read(1024):
                await out_file.write(content)

        if ml_models["image_model"] is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "A whisper model is required, es. "
                    "'nlpconnect/vit-gpt2-image-captioning'"
                ),
            )

        captioner = ml_models["image_model"]
        caption = captioner(output_dir + image.filename)
        de.load(caption[0]["generated_text"])

        return dict(filename=image.filename)


@app.post("/query/", response_model=Answer, status_code=201)
async def send_query(query: Query):
    ans = de.chat(query.text)
    return dict(question=ans.get("query"), msg=ans.get("result"))


if __name__ == "__main__":
    uvicorn.run("main:app", port=8080, log_level="info")
