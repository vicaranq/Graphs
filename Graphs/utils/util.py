
import collections


class Node:

    def __init__(self, coordinates, data):
        self.coordinates = coordinates
        self.data = data
        self.left = None
        self.right = None
        self.top = None
        self.bottom = None

        self.level = None
        self.parent = None

class Queue:
    def __init__(self):
        self.queue = collections.deque([])

    def dequeue(self):
        ''' Get left most element from the queue in O(1) time'''
        #return self.queue.pop(0) if self.queue else None
        return self.queue.popleft() if self.queue else None

    def push(self, val):
        ''' Add to end of queue in O(1) time '''
        self.queue.append(val)

    def getQueue(self):
        ''' Return current instance's queue'''
        return self.queue