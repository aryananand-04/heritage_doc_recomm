"""The one record schema every corpus document follows (decision 0004).

Every collector cleans its raw responses into these records, and the snapshot
build validates each one; the build stops on the first invalid record and names it.
"""

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DocType = Literal["site", "object", "manuscript_text", "inscription", "culture", "person_dynasty_style"]
Source = Literal["wikipedia", "met", "cleveland", "wellcome", "internet_archive", "siddham", "wikisource"]
Language = Literal["en", "hi", "kn"]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z_]+:[^\s]+$")
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    language: Language
    doc_type: DocType
    source: Source
    url: str = Field(pattern=r"^https://")
    licence: str = Field(pattern=r"^https?://")
    entity_ids: list[str] = Field(min_length=1)
    coordinates: tuple[float, float] | None = None
    period_start: int | None = None
    period_end: int | None = None
    fetched_at: datetime
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_ref: str = Field(min_length=1)

    @field_validator("entity_ids")
    @classmethod
    def wikidata_ids(cls, ids: list[str]) -> list[str]:
        bad = [q for q in ids if not (q.startswith("Q") and q[1:].isdigit())]
        if bad:
            raise ValueError(f"not Wikidata QIDs: {bad}")
        return ids

    @field_validator("coordinates")
    @classmethod
    def on_earth(cls, c):
        if c is not None and not (-90 <= c[0] <= 90 and -180 <= c[1] <= 180):
            raise ValueError(f"coordinates out of range: {c}")
        return c

    @field_validator("fetched_at")
    @classmethod
    def utc(cls, t: datetime) -> datetime:
        if t.tzinfo is None:
            raise ValueError("fetched_at must carry a time zone (UTC)")
        return t

    @model_validator(mode="after")
    def consistent(self):
        if self.sha256 != sha256(self.text):
            raise ValueError("sha256 does not match the text")
        if self.period_start is not None and self.period_end is not None and self.period_start > self.period_end:
            raise ValueError("period_start is after period_end")
        return self
