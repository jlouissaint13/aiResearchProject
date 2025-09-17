from dataclasses import dataclass

@dataclass
class Chunk:
    id: int
    text: list
    embedding: list