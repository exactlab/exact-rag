import streamlit as st
from pydantic import BaseModel
from io import StringIO
import toml
from exact_rag.dataemb import DataEmbedding
from exact_rag.config import Embeddings
from exact_rag.config import Databases
from exact_rag.image_cap import image_captioner
import os
import time
import fitz  # imports the pymupdf library
import base64

settings = toml.load("settings.toml")
output_dir = settings["server"]["output_dir"]
image_model = settings["image"].get("image_model")
settings_e = settings["embedding"]
settings_d = settings["database"]
embedding = Embeddings(**settings_e)
database = Databases(**settings_d)
de = DataEmbedding(embedding, database)


class Message(BaseModel):
    actor: str
    payload: str


def sidebar():
    FILES = "files"
    if FILES not in st.session_state:
        st.session_state[FILES] = []

    st.title("eXact-RAG")

    with st.sidebar:
        st.write("Upload text, pdf, image or audio")
        up_file = st.file_uploader(label=" ", label_visibility="collapsed")
        if up_file:
            # text
            if os.path.splitext(up_file.name)[1] == ".txt":
                if up_file.name not in st.session_state[FILES]:
                    st.session_state[FILES].append(up_file.name)
                    stringio = StringIO(up_file.getvalue().decode("utf-8"))
                    string_data = stringio.read()
                    de.load(string_data)
            # pdf
            if os.path.splitext(up_file.name)[1] == ".pdf":
                if up_file.name not in st.session_state[FILES]:
                    st.session_state[FILES].append(up_file.name)
                    with open(output_dir + up_file.name, "wb") as f:
                        f.write(up_file.getbuffer())
                        base64_pdf = base64.b64encode(up_file.getvalue()).decode(
                            "utf-8"
                        )

                    doc = fitz.open(output_dir + up_file.name)  # open a document
                    with st.spinner("Processing..."):
                        for page in doc:  # iterate the document pages
                            for annot in page.annots():
                                text = page.get_textbox(annot.rect)
                                de.load(text)
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                else:
                    base64_pdf = base64.b64encode(up_file.getvalue()).decode("utf-8")
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)

            # images
            elif os.path.splitext(up_file.name)[1] in [".jpeg", ".jpg", ".png"]:
                if up_file.name not in st.session_state[FILES]:
                    st.session_state[FILES].append(up_file.name)
                    with open(output_dir + up_file.name, "wb") as f:
                        f.write(up_file.getbuffer())
                    with st.spinner("Captioning in progress..."):
                        captioner = image_captioner(image_model)
                        caption = captioner(output_dir + up_file.name)
                    st.image(
                        output_dir + up_file.name, caption=caption[0]["generated_text"]
                    )
                    print(caption[0]["generated_text"])
                    de.load(caption[0]["generated_text"])
                else:
                    st.image(output_dir + up_file.name)
            # audio
            elif os.path.splitext(up_file.name)[1] == ".mp3":
                raise NotImplementedError
            else:
                st.error("Format not supported!")


def show():
    USER = "user"
    ASSISTANT = "ai"
    MESSAGES = "messages"
    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [
            Message(
                actor=ASSISTANT,
                payload="Hi! How can I help you?",
            )
        ]
    ### loop on state when user insert a new prompt
    ### you will not lose previous messages
    msg: Message
    for msg in st.session_state[MESSAGES]:
        st.chat_message(msg.actor).write(msg.payload)

    if prompt := st.chat_input("Ask something"):
        st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
        st.chat_message(USER).write(prompt)

        with st.chat_message(ASSISTANT):
            answer = de.chat(prompt)
            st.session_state[MESSAGES].append(
                Message(actor=ASSISTANT, payload=answer.get("result"))
            )
            res = answer.get("result")

            def _stream_data(res: str):
                for word in res.split():
                    yield word + " "
                    time.sleep(0.04)

            st.write_stream(_stream_data(res))


if __name__ == "__main__":
    sidebar()
    show()
