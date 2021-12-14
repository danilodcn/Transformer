from functools import lru_cache
from typing import Any, Dict


class Memory:

    def __init__(self) -> None:
        self.__data: Dict[str, Dict[str, Any | str]] = {}

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, value) -> dict:
        raise ValueError("Do not edit this property")

    @lru_cache(maxsize=32, typed=False)
    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.__data[key] = value

    def set_many(self, values: Dict):
        self.__data.update(values)

    def drop(self):
        self.cache_clear()
        self.__data = {}

    def cache_clear(self):
        self.get.cache_clear()
