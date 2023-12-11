from fastapi.exceptions import HTTPException
from pydantic import BaseModel, field_validator


class IntentRequest(BaseModel):
    text: str
    # NOTE: This contract catches the follwoing exception cases:
    # TEXT_EMPTY (this contract)
    # INVALID_TYPE (pydantic native)
    # TEXT_MISSING (pydantic native)
    # BODY_MISSING (pydantic native)
    # See in test/test_api.py how the excepton response schema hints on the error cause

    @field_validator("text")
    def validate_text(cls, v, values):
        # TODO: Addopt the same excpetion response schema as pydantic so that
        # all exepctions have hte same schema
        if v == "":
            detail = {
                "label": "TEXT_EMPTY",
                "message": "text field is empty",
            }
            raise HTTPException(status_code=400, detail=detail)

        return v


class IntentResponse(BaseModel):
    # NOTE: this is only one test that we do to the response.
    # See ini test/test_end_to_end.py for a more extensive list of checks that we
    # should perform on the model response. Due to limited I did not implement them here as well
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


class ReadyResponse(BaseModel):
    ready: str
