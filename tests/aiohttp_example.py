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