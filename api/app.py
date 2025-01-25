# To run:
# cd to api directory
# uvicorn app:app --reload

import json
from fastapi import FastAPI

app = FastAPI()

# FastAPI endpoint that returns the coordinates
@app.get("/get_coordinates")
async def get_coordinates():
    with open("../coordinates.json", "r") as json_file:
        coordinates = json.load(json_file)
    return coordinates
