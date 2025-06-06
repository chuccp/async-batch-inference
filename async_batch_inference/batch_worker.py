import asyncio
import logging
import multiprocessing as mp
import uuid
from typing import TypeVar, Type
from async_batch_inference.batcher import Batcher
from cacheout import Cache

log = logging.getLogger(__name__)

A = TypeVar('A', bound=Batcher)
WAIT_TIME = 0.0005


class BatchWorker:
    def __init__(self, worker_class: Type[A], batch_size: int = 16, **kwargs):

        self.batch_size = batch_size
        self.batch = []
        self.worker_class = worker_class
        self.mp = mp.get_context('spawn')
        self.worker_ready_event = self.mp.Event()
        self.result_cache = Cache(maxsize=100_000, ttl=600)
        self.send_queue = self.mp.Queue(batch_size * 2)
        self.rev_queue = self.mp.Queue(batch_size * 2)
        self.all_queue = asyncio.Queue(batch_size * 2)
        self._stop_event = asyncio.Event()
        self.kwargs = kwargs
        self.is_start = False

    async def predict(self, item: str, timeout=2.0):
        value = await asyncio.wait_for(self._write_(item), timeout=timeout)
        if isinstance(value, dict) and "error" in value:
            raise Exception(value.get("error"))
        return value

    async def _write_(self, item: str):
        queue = asyncio.Queue(1)
        await self.all_queue.put((item, queue))
        value = await queue.get()
        return value

    async def _check_send_value(self):
        while self._stop_event.is_set():
            item, queue = await self.all_queue.get()
            task_id = str(uuid.uuid4())
            self.result_cache.set(task_id, queue)
            while True:
                if self.send_queue.full():
                    await asyncio.sleep(WAIT_TIME)
                    continue
                self.send_queue.put_nowait((item, task_id))
                break

    async def _check_rev_value(self):
        while self._stop_event.is_set():
            while not self.rev_queue.empty():
                item, task_id = self.rev_queue.get_nowait()
                queue: asyncio.Queue = self.result_cache.get(task_id)
                self.result_cache.delete(task_id)
                queue.put_nowait(item)
            await asyncio.sleep(WAIT_TIME)

    async def start(self):
        if self.is_start:
            return
        self.is_start = True
        self._stop_event.set()
        worker_p: mp.context.SpawnProcess = self.mp.Process(target=self.worker_class.start,
                                                            args=(self.send_queue, self.rev_queue,
                                                                  self.worker_ready_event,
                                                                  self.batch_size, self.kwargs
                                                                  ), daemon=True)
        worker_p.start()
        is_ready = self.worker_ready_event.wait(timeout=30)
        log.info(f"==BatchWorker==start===={is_ready}=========== pid: {worker_p.pid} ")
        asyncio.create_task(self._check_send_value())
        asyncio.create_task(self._check_rev_value())

    async def stop(self):
        if not self.is_start:
            return
        self.is_start = False
        self._stop_event.clear()
        if self.worker_ready_event.is_set():
            self.worker_ready_event.clear()
