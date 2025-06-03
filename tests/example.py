from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from async_batch_inference.batch_worker import BatchWorker
from async_batch_inference.batcher import Batcher


class MyWorker(Batcher[str,str]):

    def load_model(self, **kwargs):
        print(kwargs)

    def predict_batch(self, x: list[str])->list[str]:
        return x

wrapped_model = BatchWorker(MyWorker, batch_size=16, max_delay=0.05)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await wrapped_model.start()
    yield

app = FastAPI(lifespan=lifespan)

@app.get('/batch_predict')
async def batch_predict(text: str):
    text = await wrapped_model.predict(text)
    return {"message": text}

if __name__ == '__main__':
    uvicorn.run("example:app", workers=1, host="0.0.0.0", port=1254, reload=False)