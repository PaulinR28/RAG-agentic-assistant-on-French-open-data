"""The structured answer the LLM must return, and the rules that make it trustworthy.

1. Shape: `Answer` (JSON schema sent to the model, then parsed by Pydantic).
2. Grounding: every citation must be an aid the system actually retrieved during this question,
   and a non-abstained answer must cite at least one aid. A violation raises `InvalidAnswer`,
   whose message is sent back to the model for one retry.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")  # -> "additionalProperties": false in the schema

    answer: str = Field(description="Réponse en français, concise, fondée uniquement sur les fiches fournies.")
    citations: list[str] = Field(description="Identifiants id_aid des fiches utilisées, chiffres uniquement, ex. [\"411\"].")
    abstained: bool = Field(description="true si les fiches ne permettent pas de répondre ou si la demande sort du périmètre.")

    @field_validator("citations", mode="before")
    @classmethod
    def keep_ids(cls, values: list) -> list[str]:
        """Small models write "id_aid 411" or "[411]": keep the number, drop duplicates."""
        ids = []
        for v in values or []:
            found = re.findall(r"\d+", str(v))
            ids.extend(found or [str(v)])
        return list(dict.fromkeys(ids))


ANSWER_SCHEMA = Answer.model_json_schema()


class InvalidAnswer(ValueError):
    """The reply cannot be used as is; the message explains why, in words the model can act on."""


def parse_answer(raw: str, retrieved_ids: list[str]) -> Answer:
    try:
        ans = Answer.model_validate_json(raw)
    except ValidationError as e:
        raise InvalidAnswer(f"JSON invalide ou incomplet : {e.errors()[0]['msg']}. Respecte exactement le schéma.") from e
    unknown = [c for c in ans.citations if c not in retrieved_ids]
    if unknown:
        raise InvalidAnswer(f"Citations {unknown} absentes des fiches fournies. Cite uniquement parmi : {retrieved_ids}.")
    if not ans.abstained and not ans.citations:
        raise InvalidAnswer("Une réponse non abstenue doit citer au moins un id_aid parmi les fiches fournies.")
    return ans
