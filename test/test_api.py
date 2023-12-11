from starlette.testclient import TestClient


def test_health(client: TestClient):
    response = client.get("/api/ready")

    assert response.status_code == 200


def test_intent(client: TestClient):
    """
    NOTE:Pydantic here will validate the input and output schemas so we dont have to.
    NOTE 2:The model dependency still remains. Given more time it also needs to be mocked.
    """
    response = client.post(
        "/api/intent",
        json={
            "text": "I cannot find my ultimate flight",
        },
    )

    assert response.status_code == 200


def test_request_body_missing(client: TestClient):
    response = client.post(
        "/api/intent",
    )

    assert (
        response.status_code == 422
    ), "Must throw an excepton when the body is missing"

    assert (
        response.json()["detail"][0]["type"] == "missing"
    ), "exception must indicate 'missing'"

    assert response.json()["detail"][0]["loc"] == [
        "body"
    ], "exception must indicate missing body"

    assert (
        response.json()["detail"][0]["input"] is None
    ), "exception must indicate body is None"


def test_request_missing_text_field(client: TestClient):
    response = client.post(
        "/api/intent",
        json={},
    )
    assert (
        response.status_code == 422
    ), "Must throw an exception when the text is missing"

    assert (
        response.json()["detail"][0]["type"] == "missing"
    ), "exception must indicate 'missing'"

    assert response.json()["detail"][0]["input"] == {}, "exception must indicate text"

    assert response.json()["detail"][0]["loc"] == [
        "body",
        "text",
    ], "exception must indicate missing field"



def test_request_missing_invalid_text_type(client: TestClient):
    response = client.post(
        "/api/intent",
        json={"text": 4},
    )
    assert (
        response.status_code == 422
    ), "Must throw an exception when the text is missing"

    assert (
        response.json()["detail"][0]["type"] == "string_type"
    ), "exception must indicate 'string_type'"
    assert (
        response.json()["detail"][0]["input"] == 4
    ), "exception must include the bad input value"

    assert response.json()["detail"][0]["loc"] == [
        "body",
        "text",
    ], "exception must indicate missing field"


