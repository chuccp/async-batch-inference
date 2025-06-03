import logging
import traceback
from typing import TypeVar, Generic

log = logging.getLogger(__name__)
X = TypeVar('X')
Y = TypeVar('Y')


class Batcher(Generic[X, Y]):

    def __init__(self, send_queue, rev_queue, worker_ready_event, batch_size, kwargs):
        self.send_queue = send_queue
        self.rev_queue = rev_queue
        self.worker_ready_event = worker_ready_event
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.max_size = batch_size
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        """
        加载模型
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def predict_batch(self, x: list[X])->list[Y]:
        """
        批量预测
        :param x:
        :return:
        """
        raise NotImplementedError()

    def _run_task(self):
        while True:
            try:
                value_list: list[str] = []
                while True:
                    value = self.send_queue.get()
                    value_list.append(value)
                    if self.send_queue.empty() or len(value_list) >= self.max_size:
                        input_list = [(v[0], v[1]) for v in value_list]
                        try:
                            v_list = self.predict_batch([v[0] for v in input_list])
                            for item1, item2 in zip(v_list, input_list):
                                self.rev_queue.put((item1, item2[1]))
                        except Exception as e:
                            stack_trace = traceback.format_exc()
                            log.error(f"_run_task========2 error {stack_trace}")
                            for item2 in input_list:
                                self.rev_queue.put(({"error": e.__str__()}, item2[1]))
                        break
            except Exception as e:
                stack_trace = traceback.format_exc()
                log.error(f"_run_task========1 e {e} error {stack_trace}")

    def _run(self):
        self._run_task()

    @classmethod
    def start(cls, send_queue, rev_queue, worker_ready_event, batch_size, kwargs):
        w = cls(send_queue, rev_queue, worker_ready_event, batch_size, kwargs)
        worker_ready_event.set()
        w._run()
