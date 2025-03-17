# Event Handler Interface
from abc import abstractmethod, ABC


class EventHandler(ABC):
    @abstractmethod
    def handle(self):
        pass