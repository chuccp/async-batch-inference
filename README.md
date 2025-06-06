# async-batch-inference

`async-batch-inference` 是一个Python库，它支持异步批量推理，通过并发处理多个请求来提高效率。

## 项目背景
在进行模型推理时，批量处理请求能够显著提升吞吐量和资源利用率。`async-batch-inference` 提供了一个简单易用的接口，允许开发者在不牺牲响应时间的前提下，实现异步批量推理。

## 安装步骤
### 克隆仓库
```bash
  git clone https://github.com/your-repo/async-batch-inference.git
  cd async-batch-inference
```


```bash
  pip install -r requirements.txt
```

```bash
  python setup.py install
```

## 使用示例
### fastapi基本使用
```python
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from async_batch_inference.batch_worker import BatchWorker
from async_batch_inference.batcher import Batcher
class MyWorker(Batcher[str,str]):

    def load_model(self, **kwargs):
        """
        加载模型
        :param kwargs:
        :return:
        """
        print(kwargs.get("model_path"))

    def predict_batch(self, x: list[str])->list[str]:
        return ["text:" + str(i) for i in x]

wrapped_model = BatchWorker(MyWorker, batch_size=16,  model_path = 'model_path')


@asynccontextmanager
async def lifespan(app: FastAPI):
     # 在这里添加需要在后台运行的任务
    await wrapped_model.start()
    yield

app = FastAPI(lifespan=lifespan)

@app.get('/predict')
async def predict(text: str):
    text = await wrapped_model.predict(text)
    return {"message": text}

if __name__ == '__main__':
    uvicorn.run("fastapi_example:app", workers=1, host="0.0.0.0", port=1254, reload=False)
```


### aiohttp 基本使用
```python
from aiohttp import web
from async_batch_inference.batch_worker import BatchWorker
from async_batch_inference.batcher import Batcher


class MyWorker(Batcher[str,str]):

    def load_model(self, **kwargs):
        """
        加载模型
        :param kwargs:
        :return:
        """
        print(kwargs.get("model_path"))

    def predict_batch(self, x: list[str])->list[str]:
        return ["text:" + str(i) for i in x]

wrapped_model = BatchWorker(MyWorker, batch_size=16,  model_path = 'model_path')

async def start_background_tasks(_app_):
    # 在这里添加需要在后台运行的任务
    await wrapped_model.start()



# 请求处理函数
async def predict(request):
    text: str = request.query.get("text")
    text = await wrapped_model.predict(text)
    return web.Response(text=text)

# 创建应用并添加路由
app = web.Application()
app.on_startup.append(start_background_tasks)
app.router.add_get('/predict', predict)  # 处理 GET 请求

# 启动服务器
if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=1254)
```

## 参考来源
- [InferLight](https://github.com/thuwyh/InferLight)