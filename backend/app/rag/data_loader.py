import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from backend.app.utils.text import clean_text, join_lines
from backend.app.rag.types import DocumentChunk


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _flatten(value: Any, prefix: str = "") -> Dict[str, str]:
    items: Dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(_flatten(item, next_prefix))
        return items
    if isinstance(value, list):
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            items.update(_flatten(item, next_prefix))
        return items
    if value is None:
        return {}
    items[prefix] = clean_text(str(value))
    return items


def _record_to_text(record: Dict[str, Any]) -> str:
    flattened = _flatten(record)
    lines = [f"{key}: {value}" for key, value in flattened.items() if value]
    return join_lines(lines)


def _iter_records(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(value, dict):
                yield value


def load_documents(data_dir: Path, max_text_chars: int) -> List[DocumentChunk]:
    documents: List[DocumentChunk] = []
    for path in sorted(data_dir.glob("*.json")):
        data = load_json(path)
        for record in _iter_records(data):
            text = _record_to_text(record)
            if not text:
                continue
            if len(text) > max_text_chars:
                text = text[:max_text_chars]
            patient_id = record.get("patientId") if isinstance(record, dict) else None
            timestamp = record.get("timestamp") if isinstance(record, dict) else None
            documents.append(
                DocumentChunk(
                    source=path.name,
                    patient_id=patient_id,
                    timestamp=timestamp,
                    text=text,
                    metadata={"source": path.name},
                )
            )
    return documents
