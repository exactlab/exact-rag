from typing import Any
from pydantic import BaseModel


class FromDict:
    def __init__(self, *path: str):
        self._path = list(path)

    def __call__(self, d: dict[str, Any]):
        def unwrap(d: dict[str, Any], key: str):
            if isinstance(d, dict):
                return d.get(key)
            else:
                return None

        current_dict = d
        for key in self._path:
            current_dict = unwrap(current_dict, key)

        return current_dict


class Settings(BaseModel):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        output_dict = dict()
        for var, field_info in self.model_fields.items():
            from_dicts = [
                metadata
                for metadata in field_info.metadata
                if isinstance(metadata, FromDict)
            ]
            if len(from_dicts) == 0:
                output_dict[var] = kwargs.get(var)
            elif len(from_dicts) == 1:
                output_dict[var] = from_dicts[0](kwargs)
            else:
                raise TypeError(f"Multiple FromDict in {var} annotation.")

        super(Settings, self).__init__(**output_dict)
