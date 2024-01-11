import requests
from rich import print

BASE_URL = "http://0.0.0.0:8080"

def upload_file(file_path, endpoint="/upload/"):
    with open(file_path, "rb") as file:
        files = {"file": file}
        response = requests.post(f"{BASE_URL}{endpoint}", files=files)
        return response.json()

def send_query(query, endpoint="/query/"):
    data = {"query": query}
    response = requests.post(f"{BASE_URL}{endpoint}", json=data)
    return response.json()

if __name__ == "__main__":
    # Example: Upload a file
    file_response = upload_file("test.txt")
    print("[bold green]File Upload Response:[/bold green]")
    print(file_response)

    # Example: Send a text query
    text_query = "Hello, FastAPI!"
    query_response = send_query(text_query)
    print("\n[bold blue]Text Query Response:[/bold blue]")
    print(query_response)
