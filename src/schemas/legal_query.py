from pydantic import BaseModel, Field


class LegalQueryRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=2,
        description="Legal question in Arabic or English",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )