from pydantic import BaseModel, Field

from .sources import LegalSource


class LegalResponse(BaseModel):
    """
    Response returned by the Arabic Legal Document RAG API.
    """

    answer_ar: str = Field(
        description=(
            "The answer to the user's legal question in Arabic. "
            "The answer must be based only on the retrieved legal sources."
        )
    )

    sources: list[LegalSource] = Field(
        default_factory=list,
        description=(
            "Legal articles retrieved and used to support the answer. "
            "Sources are identified by article citation."
        ),
    )

    cannot_answer: bool = Field(
        default=False,
        description=(
            "Whether the system cannot provide a reliable answer "
            "from the retrieved legal sources."
        ),
    )