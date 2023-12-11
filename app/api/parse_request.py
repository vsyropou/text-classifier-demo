# from app.api.api import IntentRequest
from fastapi.exceptions import HTTPException


def parse_request(request):
    # NOTE: Most of tehese excpetion are handled by pydantic. Regardless I still implemnted them

    # catch request does not have body

    try:
        assert request.model_dump_json()
    except AssertionError as err:
        detail = {"label": "BODY_MISSING", "message": "Request does not have a body"}

        raise HTTPException(status_code=400, detail=detail) from err
