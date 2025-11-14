from dataclasses import dataclass

@dataclass
class Chunk:
    id: str # lets make this the same hashvalue that is used for the metadata
    text: list
    embedding: list
