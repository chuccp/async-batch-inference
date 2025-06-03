from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from batch_inference.batch_worker import BatchWorker
from batch_inference.batcher import Batcher, T


class MyWorker(Batcher[str]):



    def load_model(self, **kwargs):
        print(kwargs)

    def predict_batch(self, x: list[str]):
        return x

wrapped_model = BatchWorker(MyWorker, batch_size=16, max_delay=0.05)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await wrapped_model.start()
    yield

app = FastAPI(lifespan=lifespan)

@app.get('/batch_predict')
async def batch_predict():
    return {"message": "Hello World"}

if __name__ == '__main__':
    uvicorn.run("example:app", workers=1, host="0.0.0.0", port=1254, reload=False)