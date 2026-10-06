from pydantic import BaseModel, Field


class LegalSource(BaseModel):
    """
    Represents one legal article retrieved from the Egyptian Civil Code.
    """

    article_number: int = Field(
        description="The article number in the Egyptian Civil Code.",
    )

    book: str | None = Field(
        default=None,
        description="The book containing the article.",
    )

    chapter: str | None = Field(
        default=None,
        description="The chapter containing the article.",
    )

    section: str | None = Field(
        default=None,
        description="The section containing the article.",
    )

    topic: str | None = Field(
        default=None,
        description="The topic or subsection containing the article.",
    )

    text_ar: str | None = Field(
        default=None,
        description="The original Arabic legal text of the article, if available.",
    )

    text_en: str | None = Field(
        default=None,
        description="The English translation of the article, if available.",
    )

    is_repealed: bool = Field(
        default=False,
        description="Whether this article has been repealed.",
    )

    source_page: int | None = Field(
        default=None,
        description="The PDF page where this article appears.",
    )

    citation: str = Field(
        description="The formal citation of the legal article.",
    )

    rerank_score: float | None = Field(
        default=None,
        description="The score assigned by the reranker.",
    )