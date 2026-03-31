from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataItem:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    # Optional fields for generation during validation
    prompt_input_ids: Optional[List[int]]
    prompt_attention_mask: Optional[List[int]]
    prompt_length: Optional[int]
    prompt_text: Optional[str]
    target_text: Optional[str]
    sample_id: Optional[int]

    def __getitem__(self, key: str):
        """Allow dict-style access while enforcing real dataclass fields."""
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value):
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
            return
        raise KeyError(key)

    def get(self, key: str, default=None):
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        return default

    @classmethod
    def keys(cls):
        return cls.__dataclass_fields__.keys()
