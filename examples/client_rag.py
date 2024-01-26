import requests


BASE_URL = "http://0.0.0.0:8080"


def upload_file(file_path, endpoint="/upload_text/"):
    with open(file_path, "rb") as file:
        files = {"file": file}
        response = requests.post(f"{BASE_URL}{endpoint}", files=files)
        return response.json()


def upload_audio(file_path, endpoint="/upload_audio/"):
    files = {"audio": (f"{file_path}", open(f"{file_path}", "rb"), "audio/mp3")}
    response = requests.post(f"{BASE_URL}{endpoint}", files=files)
    return response.json()


def upload_image(file_path, endpoint="/upload_image/"):
    files = {"image": (file_path, open(file_path, "rb"), "image/png")}
    response = requests.post(f"{BASE_URL}{endpoint}", files=files)
    return response.json()


def send_query(query, endpoint="/query/"):
    data = {"text": query}
    response = requests.post(f"{BASE_URL}{endpoint}", json=data)
    return response.json()


if __name__ == "__main__":
    # Example: Upload a file
    file_response = upload_file("test.txt")

    # file_response = upload_audio("test.mp3")

    # file_response = upload_image("max.jpeg")
    print(file_response)

    text_query = "Tell me something"
    query_response = send_query(text_query)
    print(query_response)
