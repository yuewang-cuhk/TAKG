import numpy as np

class SourceRepresentationQueue:
    # A FIFO memory for storing the encoder representations
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = []
        self.position = 0

    def put(self, tensor):
        if len(self.queue) < self.capacity:
            self.queue.append(None)
        self.queue[self.position] = tensor
        self.position = (self.position + 1) % self.capacity

    def sample(self, sample_size):
        if len(self.queue) < sample_size:
            return None
        # return random.sample(self.queue, sample_size)
        idxs = np.random.choice(len(self.queue), sample_size, replace=False)
        return [self.queue[i] for i in idxs]

    def __len__(self):
        return len(self.queue)
