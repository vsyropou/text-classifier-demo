"""
Our Ultimate API:

Main endpoints;
 POST: /api/intent:  It predicts the monetary value of apassenger in the next 30 days

Request and response schemas can be found in app.api_contracnts.py

For detailed endpoint documentation run the app and visit the url: http://localhost:8080/docs

Here are some request examples for the command line:

predict endpoit:

    curl
        -d '{

            }'
        -H "Content-Type: applicatiion/json"
        -X
        POST http://localhost:8080/api/predict

requests endpoint:
    curl  http://localhost:8080/api/requests/1
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.contracts import IntentRequest, IntentResponse
from app.api.load_model import load_model, load_tags

app = FastAPI()


models = {}
model_tags = {}


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    This context is triggered before the server is up and
    stays open for as longs as the server is up
    """
    # Load the ML model, the methods are can handle exceptions
    # TODO: Choose top n from docker manifest file
    models["intent_classifier"] = load_model(top_n=3)
    model_tags["intent_classifier_tags"] = load_tags()

    yield

    models.clear()
    model_tags.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/api/intent", response_model=IntentResponse)
async def intent(
    request_payload: IntentRequest,
) -> IntentResponse:
    """
    Predicts the intent of a given text query
    """

    # NOTE: Here We would want to log a few things asynchronously:
    # 1) log invocations input and output to a table to avoid feature disparity in training
    # 2) log the invocation time and latency and log it to a table
    # 3) log a request  has and the model tags for tracability of the request

    # timestamp = time.time()
    # trace_json = json.dumps({**request_payload.model_dump(), "timestamp": str(timestamp)})
    # trace_hash = hashlib.md5(trace_json.encode("utf-8")).hexdigest()

    # TODO: async log input
    # invoke the model
    response_model = models["intent_classifier"](model_input=request_payload.text)

    # TODO: async log output
    # latency = str((time.time() - timestamp) * 1000)

    return IntentResponse(intents=response_model)
