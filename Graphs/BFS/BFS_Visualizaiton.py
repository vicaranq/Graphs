import ast
import matplotlib.pyplot as plt
from  matplotlib.pyplot import figure, draw, pause
import matplotlib
import numpy as np
import time
import sys

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
        self.queue = []

    def push(self,val):
        self.queue.append(val)
    def dequeue(self):
        return self.queue.pop(0) if self.queue else None
    def getQueue(self):
        return self.queue

def linkNodeToChildren(node, numRows, numCols, input_map):
    '''
    Link nodes to node in map with the proper level
    :return:
    '''

    new_level = node.level + 1
    x, y = node.coordinates

    # link to top/bottom/left/right if it is within boundaries and equal to 1
    linkNodeHelper(x-1,  y, numRows, numCols, input_map, new_level,node, 'top')
    linkNodeHelper(x, y+1, numRows, numCols, input_map, new_level, node, 'right')
    linkNodeHelper(x+1,  y, numRows, numCols, input_map, new_level,node, 'bottom')
    linkNodeHelper(x, y-1, numRows, numCols, input_map, new_level,node, 'left')


def linkNodeHelper( x, y, numRows, numCols, input_map, new_level,node, attrname):

    # link to top if it is within boundaries and if top is 1
    if 0 <= x and x < numRows \
            and 0 <= y and y < numCols \
            and (input_map[x][y] == 1 or input_map[x][y] == 9):   # check rows boundary and top equal to 1
        # Create top node
        side_node = Node([x,y],input_map[x][y])
        side_node.level = new_level
        side_node.parent = node
        setattr(node, attrname, side_node)

def create_img(img, numRows, numCols, input_map, found_path_flag=False):
    for i in range(numRows):
        for j in range(numCols):
            if not found_path_flag:
                if input_map[i][j] == 1:
                    img[i][j] = [192/250,192/250,192/250] # dark grey
                elif input_map[i][j] == 9:
                    img[i][j] = [50/250,250/250,50/250]   # green
                elif input_map[i][j] == -1:
                    img[i][j] = [234/250,242/250,128/250]
                else:
                    img[i][j] = [0,0,0]
            else:
                if input_map[i][j] == -1 or input_map[i][j] == 9:
                    img[i][j] = [122/250,130/250,239/250] # dark grey
                else:
                    img[i][j] = [0,0,0]

    return img

def paint_path(node,img, img_artist, fg):

    if node:

        x,y = node.coordinates
        img[x][y] = [122/250,130/250,239/250] # dark grey
        img_artist.set_data(img)
        fg.canvas.draw()
        fg.canvas.flush_events()

        paint_path(node.parent,img, img_artist, fg)

    else:
        plt.pause(2)



def apply_BFS(numRows, numCols, input_map):

    # Initialize root node
    data = input_map[0][0]
    root = Node([0,0], data)
    root.level = 0

    # Initialize queue
    queue = Queue()
    queue.push(root)

    # Initialize image
    img = np.zeros((numRows,numCols,3))
    img = create_img(img, numRows, numCols, input_map)

    matplotlib.use('TkAgg')
    fg = figure()
    axs = plt.gca()
    img_artist = axs.imshow(img)
    plt.show(block=False)
    plt.title('Applying BFS on a '+str(numRows)+'x'+str(numCols)+' Matrix')
    plt.xlabel('Current Level Depth: 0 ')
    background = fg.canvas.copy_from_bbox(axs.bbox)

    plt.pause(1)

    while queue.queue:
        # Get node
        node = queue.dequeue()

        if node.data == 9:
            plt.xlabel('Target Level Depth: '+ str(node.level))
            paint_path(node,img, img_artist, fg)
            return node.level

        # link to  children
        linkNodeToChildren(node, numRows, numCols, input_map) # if not node.linked: l

        # if children at top/right/... then add to queue
        if node.top:
            queue.push(node.top)
        if node.right:
            queue.push(node.right)
        if node.bottom:
            queue.push(node.bottom)
        if node.left:
            queue.push(node.left)

        x,y = node.coordinates
        input_map[x][y] = -1

        img[x][y] = [234/250,242/250,128/250]
        img_artist.set_data(img)
        #plt.xlabel('Current Level Depth: '+ str(node.level))
        fg.canvas.restore_region(background)
        axs.set_xlabel('Current Level Depth: '+ str(node.level))
        axs.draw_artist(img_artist)
        fg.canvas.blit(axs.bbox)


    return -1


def get_distance(numRows, numCols, input_map):

    # 1. Create graph of nodes containing 1 until 9 is found, else return 9.
    min_distance = apply_BFS(numRows, numCols, input_map)

    return min_distance

def get_map(m,n):

    X = np.random.rand(m,n)

    X[X>=0.5] = 1
    X[X< 0.5] = 0
    X[0, :] = np.ones(n)
    X[:,-1] = np.ones(m)
    X[-1,-1] = 9

    return X

if __name__ == '__main__':

    # 3x3 case
    numRows = 3 #int(input())
    numCols = 3 #int(input())
    input_map = [[1,0,0], [1,0,0], [1,9,0]]#ast.literal_eval(input())
    print(get_distance(numRows, numCols, input_map))

    # 5x5
    numRows = 5 #int(input())
    numCols = 5 #int(input())
    input_map = [[1,0,0,1,9],
                 [1,0,0,1,0],
                 [1,1,0,1,1],
                 [1,1,0,0,1],
                 [1,1,1,1,1]]
    print(get_distance(numRows, numCols, input_map))

    # 8x8
    numRows = 8 #int(input())
    numCols = 8 #int(input())
    input_map = [[1, 1, 1, 1, 1, 1, 0, 0],
                 [1, 0, 0, 1, 0, 1, 1, 0],
                 [1, 0, 0, 1, 0, 0, 1, 0],
                 [1, 0, 0, 1, 1, 1, 1, 0],
                 [1, 0, 0, 1, 0, 0, 1, 0],
                 [1, 1, 1, 1, 0, 1, 1, 0],
                 [1, 1, 0, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 0, 0, 0, 9]]

    print(get_distance(numRows, numCols, input_map))


    # mxn
    m = 50
    n = 50
    X = get_map(m,n)

    print(get_distance(m, n, X))