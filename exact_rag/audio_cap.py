from whisper.model import transcribe_function


def audio_caption(audio_file: str) -> str:
    res = transcribe_function(audio_file)
    return res
