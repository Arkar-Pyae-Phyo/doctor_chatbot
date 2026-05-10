import re
from typing import Iterable


def clean_text(value: str) -> str:
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def join_lines(lines: Iterable[str]) -> str:
    return "\n".join([line for line in lines if line])
