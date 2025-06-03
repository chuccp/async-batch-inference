import asyncio
import multiprocessing as mp
import uuid
from typing import TypeVar, Type
from async_batch_inference.batcher import Batcher
from cacheout import Cache

A = TypeVar('A', bound=Batcher)
week_time = 0.0005


class BatchWorker:
    def __init__(self, worker_class: Type[A], batch_size: int = 16, **kwargs):
        self.batch_size = batch_size
        self.batch = []
        self.worker_class = worker_class
        self.mp = mp.get_context('spawn')
        self.result_cache = Cache(maxsize=100_000, ttl=600)
        self.send_queue = self.mp.Queue(batch_size)
        self.rev_queue = self.mp.Queue(batch_size)
        self.all_queue = asyncio.Queue(batch_size)
        self.kwargs = kwargs
        self.is_start = False

    async def predict(self, item: str, timeout=2.0):
        return await asyncio.wait_for(self._write_(item), timeout=timeout)

    async def _write_(self, item: str):
        queue = asyncio.Queue(1)
        await self.all_queue.put((item, queue))
        value = await queue.get()
        return value

    async def _check_send_value(self):
        while True:
            item, queue = await self.all_queue.get()
            task_id = str(uuid.uuid4())
            self.result_cache.set(task_id, queue)
            while True:
                if self.send_queue.full():
                    await asyncio.sleep(week_time)
                    continue
                self.send_queue.put_nowait((item, task_id))
                break

    async def _check_rev_value(self):
        while True:
            while not self.rev_queue.empty():
                item, task_id = self.rev_queue.get_nowait()
                queue: asyncio.Queue = self.result_cache.get(task_id)
                self.result_cache.delete(task_id)
                queue.put_nowait(item)
            await asyncio.sleep(week_time)

    async def start(self):
        if self.is_start:
            return
        self.is_start = True
        worker_ready_event = self.mp.Event()
        worker_p: mp.context.SpawnProcess = self.mp.Process(target=self.worker_class.start,
                                                            args=(self.send_queue, self.rev_queue, worker_ready_event,
                                                                  self.batch_size, self.kwargs
                                                                  ), daemon=True)
        worker_p.start()
        is_ready = worker_ready_event.wait(timeout=30)
        print(f"==BatchWorker==start===={is_ready}=========== pid: {worker_p.pid} ")
        asyncio.create_task(self._check_send_value())
        asyncio.create_task(self._check_rev_value())
