
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

class PriorityQueue:
    def __init__(self):
        self.queue = collections.deque([])
        self.priority_queue = collections.deque([])

    def dequeue(self):
        ''' Get left most element from the priority queue in O(1) time '''
        # Dequeue from Priority queue if it exists
        if self.priority_queue:
            return self.priority_queue.popleft()
        return self.queue.popleft() if self.queue else None

    def push(self, val, priority = 0):
        '''

        :param val: Value to push in priority queue
        :param priority: int referring to the priority {0: normal push, 1: priority)
        :return: None
        '''
        if priority:
            self.priorityPush(val)
        else:
            self.normal_push(val)

    def normal_push(self, val):
        ''' Add to end of queue in O(1) time '''
        self.queue.append(val)

    def priorityPush(self, val):
        ''' Add to end of priority queue in O(1) time '''
        self.priority_queue.append(val)

    def getQueue(self):
        ''' Return current instance's queue [priority queue extended to queue]'''
        priority_copy = [node.coordinates for node in self.priority_queue]
        priority_copy.extend([node.coordinates for node in self.queue])
        return priority_copy
