import multiprocessing as mp
import queue
import time
import traceback
from typing import TypeVar, Generic, Any

X = TypeVar('X')
Y = TypeVar('Y')


class Batcher(Generic[X, Y]):

    def __init__(self, send_queue: mp.Queue, rev_queue: mp.Queue, worker_ready_event,
                 batch_size, kwargs):
        self.send_queue = send_queue
        self.rev_queue = rev_queue
        self.worker_ready_event = worker_ready_event
        self.kwargs = kwargs
        self.max_size = batch_size
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        """
        加载模型
        :param kwargs:
        :return:
        """
        pass

    def predict_batch(self, x: list[X]) -> list[Y]:
        """
        批量预测
        :param x:
        :return:
        """
        raise NotImplementedError()

    def _predict(self, batch_data: list[tuple[X, str]]) -> list[tuple[Y, str]]:
        try:
            v_list = self.predict_batch([v[0] for v in batch_data])
            return [(v, v1[1]) for v, v1 in zip(v_list, batch_data)]
        except Exception as e:
            stack_trace = traceback.format_exc()
            return [({"error": stack_trace}, v1[1]) for v1 in batch_data]

    def _run_task(self):
        while self.worker_ready_event.is_set():
            batch_data: list[tuple[X, str]] = []
            try:
                first_value = self.send_queue.get(timeout=1)
                batch_data.append(first_value)
                while len(batch_data) < self.max_size:
                    try:
                        next_value = self.send_queue.get_nowait()
                        batch_data.append(next_value)
                    except queue.Empty:
                        break
            except queue.Empty:
                pass
            except Exception as e:
                time.sleep(0.1)

            if len(batch_data) > 0:
                output_list = self._predict(batch_data)
                for item in output_list:
                    self.rev_queue.put(item)

    def _run(self):
        self._run_task()

    @classmethod
    def start(cls, send_queue, rev_queue, worker_ready_event, batch_size, kwargs):
        w = cls(send_queue, rev_queue, worker_ready_event, batch_size, kwargs)
        worker_ready_event.set()
        w._run()
