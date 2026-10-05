from typing import List

from pydantic import BaseModel, Field

from .sources import LegalSource


class LegalResponse(BaseModel):
    """
    Response returned by the Arabic Legal RAG /ask endpoint.
    """

    answer_ar: str = Field(
        description=(
            "The answer to the user's legal question in Arabic. "
            "The answer must be based only on the retrieved legal sources."
        )
    )

    sources: List[LegalSource] = Field(
        default_factory=list,
        description=(
            "The legal articles retrieved and used to support the answer. "
            "May contain one or multiple articles."
        )
    )

    cannot_answer: bool = Field(
        default=False,
        description=(
            "Whether the system cannot provide a reliable answer "
            "from the retrieved legal sources."
        )
    )