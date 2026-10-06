from typing import Optional

from pydantic import BaseModel


class LegalSource(BaseModel):
    article_number: int
    book: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    topic: Optional[str] = None

    text_ar: Optional[str] = None
    text_en: Optional[str] = None

    is_repealed: bool = False
    source_page: Optional[int] = None
    citation: str
    source_status: Optional[str] = None