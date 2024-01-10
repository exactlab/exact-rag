from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from fastapi import HTTPException
from pydantic import BaseModel

app = FastAPI()



@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".txt"):
        contents = await file.read()
        return {"filename": file.filename, "contents": contents.decode("utf-8"), "file_type": "txt"}
    
    elif file.filename.endswith(".json"):
        contents = await file.read()
        return {"filename": file.filename, "contents": contents.decode("utf-8"), "file_type": "json"}
    
    else:
        raise HTTPException(status_code=400, detail="Only .txt and .json files are allowed")

@app.post("/query/")
async def send_query(query: str):
    
    return {"query": query}