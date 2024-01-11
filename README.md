Run the application:

```bash
poetry run uvicorn main:app --host 0.0.0.0 --post 8080
```

On another shell:

```bash
poetry run exact_rag/client_rag.py
```
