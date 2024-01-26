from pydantic import BaseModel


class Query(BaseModel):
    text: str


class GenericResponse(BaseModel):
    filename: str


class Answer(BaseModel):
    question: str
    msg: str
