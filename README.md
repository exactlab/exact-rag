# Exact-RAG: Multimodal Retrieval-Augmented Generation :dart:

Exact-RAG is a powerful multimodal model designed for Retrieval-Augmented Generation (RAG). It seamlessly integrates text, visual and audio information, allowing for enhanced content understanding and generation. This repository contains the source code, and example scripts to facilitate the usage and exploration of Exact-RAG.

## Prerequisites

* Python >= 3.10
* Poetry ([Install](https://python-poetry.org/docs/))

#### LLMs
* To use OpenAI models you need a valid key at this [link](https://platform.openai.com/api-key)
* To use local LLM models you need Ollama. [Here](https://ollama.ai/download) the instructions to install

#### Ollama
You should download a LLM model usin ollama. By default the `orca2` model is set in `settings.toml`, to download just run:

```bash
ollama run orca2
```

#### Databases
* To use Elasticsearch you should have a running cluster or you can use a test deployment using the [docker-es.sh](./scripts/docker-es.sh) script.


## Installation

Clone the repository:

```bash
poetry install # -E audio -E image
```

* audio extra will install `openai-whisper` for speech-to-text

* image extra will install `transformers` and `pillow` for image captioning

## Usage - Server

First step is to modify the `settings.toml` file.

Then starting the web server just running:

```bash
poetry run python exact_rag/main.py
```

> NOTE: The first start up could required some time to download the selected models, expecially for image captioning

## Usage - UI (Demo purpose)

UI Demo is build upon `streamlit` and it is made just to demo purposes.
If you want to run locally to quick try eXact-RAG features, be sure to have install the packege with `dev` dependencies, and then:

```bash
poetry run streamlit run frontend/ui.py
```

## Examples

You can find some examples of usage in the examples [folder](./examples/)

Chat with __images__ example:

https://github.com/exactlab/exact-rag/assets/43796979/d8daaa12-664b-4a38-9959-e79f752a03b1

Chat with __PDF__ and __Tables__:

https://github.com/exactlab/exact-rag/assets/43796979/dd5c6737-99bf-4fa1-b30e-1d3e0d22b2d7



## Tests

To run the tests:
* be sure to have install the `dev` dependecies with:
    ```bash
    poetry install --with-dev # -E audio -E image
    ```
* then run:
    ```bash
    poetry run pytest tests/
    ```

## Contributing

We welcome contributions! If you'd like to contribute to Exact-RAG, please follow our contribution [guidelines](CONTRIBUTING.md).


* Fork the repo.
* Clone the repo from your codebase
* Choose your favorite editor and open the folder.
* Create a new branch: git checkout -b  <branch-name>
* Make changes, commit them and push it back up to github using git  push origin <your-branch-name>.
* Open pull request on GitHub.


## License

This project is licensed under the MIT License.
