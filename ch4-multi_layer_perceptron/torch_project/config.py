from dataclasses import dataclass


@dataclass
class Config:
    device: str
    EPOCHS: int
    BATCH_SIZE: int
    LEARNING_RATE: float
