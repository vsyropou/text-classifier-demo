from fastapi.exceptions import HTTPException
from pydantic import BaseModel, field_validator
from pydantic.fields import Field


class IntentRequest(BaseModel):
    text: str = Field(min_length=1)


class IntentResponse(BaseModel):
    # NOTE: this is only one test that we do to the response.
    # See ini test/test_end_to_end.py for a more extensive list of checks that we
    # should perform on the model response.
    intents: list[dict[str, str | float]]

    @field_validator("intents")
    def validate_intents(cls, v, values):
        if type(v) is not list:
            detail = {
                "label": "INVALID_TYPE",
                "message": f"bad model response: {v}",
            }
            raise HTTPException(status_code=400, detail=detail)

        return v
