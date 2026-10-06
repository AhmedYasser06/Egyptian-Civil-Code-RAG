from pydantic import BaseModel, Field


class AskResponse(BaseModel):
    answer: str = Field(
        description="Answer to the legal question in Arabic."
    )

    sources: list[str] = Field(
        default_factory=list,
        description="Egyptian Civil Code article citations supporting the answer.",
    )