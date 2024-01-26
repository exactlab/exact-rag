# Examples

> Be sure to have ollama running (localhost://11434 -> Ollama is running)

> Be sure to have configured properly the settings.toml file

To run this example you had to activate the web server:
```bash
poetry run python exact_rag/main.py
```

Then in another shell you can run the `client_rag.py` which provides a simple client to send data to RAG and chatting.

```bash
poetry run python client_rag.py
```
