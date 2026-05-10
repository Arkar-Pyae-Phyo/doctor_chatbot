from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DocumentChunk:
    source: str
    patient_id: Optional[str]
    timestamp: Optional[str]
    text: str
    metadata: Dict[str, str]


@dataclass
class RetrievedChunk:
    source: str
    patient_id: Optional[str]
    timestamp: Optional[str]
    text: str
    score: float
    metadata: Dict[str, str]
