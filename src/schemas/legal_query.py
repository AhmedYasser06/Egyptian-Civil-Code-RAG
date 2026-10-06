from pydantic import BaseModel, Field, field_validator


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

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if len(value) < 2:
            raise ValueError("Question must contain at least 2 non-whitespace characters.")

        return value