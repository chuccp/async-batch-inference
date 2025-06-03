import logging
import traceback
from typing import TypeVar, Generic
log = logging.getLogger(__name__)
T = TypeVar('T')


class Batcher(Generic[T]):

    def __init__(self, send_queue, rev_queue, worker_ready_event, batch_size, kwargs):
        self.send_queue = send_queue
        self.rev_queue = rev_queue
        self.worker_ready_event = worker_ready_event
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.max_size = batch_size

    def load_model(self, **kwargs):
        raise NotImplementedError()

    def predict_batch(self, x: list[T]):
        raise NotImplementedError()

    def _run_task(self):
        value_list: list[str] = []
        while True:
            try:
                value = self.send_queue.get()
                value_list.append(value)
                if self.send_queue.empty() or len(value_list) >= self.max_size:
                    input_list = [(v[0], v[1]) for v in value_list]
                    value_list = []
                    try:
                        v_list = self.predict_batch([v[0] for v in input_list])
                        for item1, item2 in zip(v_list, input_list):
                            self.rev_queue.put((item1, item2[1]))
                    except Exception as e:
                        stack_trace = traceback.format_exc()
                        log.error(f"_run_task========2 error {stack_trace}")
                        for item2 in input_list:
                            self.rev_queue.put(({"error": e.__str__()}, item2[1]))
            except Exception as e:
                stack_trace = traceback.format_exc()
                log.error(f"_run_task=======1 error {e}")
                stack_trace = traceback.format_exc()
                log.error(f"_run_task========1 error {stack_trace}")

    def _run(self):
        self._run_task()

    @classmethod
    def start(cls, send_queue, rev_queue, worker_ready_event, batch_size, kwargs):
        w = cls(send_queue, rev_queue, worker_ready_event, batch_size, kwargs)
        worker_ready_event.set()
        w._run()

        # self.load_model(**kwargs)
        # worker_ready_event.set()
        # value_list: list[str] = []
        # while True:
        #     value = send_queue.get()
        #     value_list.append(value)
        #     if send_queue.empty() or len(value_list) >= self.max_size:
