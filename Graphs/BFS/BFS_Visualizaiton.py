import matplotlib.pyplot as plt
from  matplotlib.pyplot import figure
import matplotlib
import numpy as np
from utils.util import Node, Queue
import time


def linkNodeToChildren(node, numRows, numCols, input_map):
    '''
    Link nodes to node in map with the proper level
    '''

    new_level = node.level + 1
    x, y = node.coordinates

    # link to top/bottom/left/right if it is within boundaries and equal to 1
    linkNodeHelper(x-1,  y, numRows, numCols, input_map, new_level,node, 'top')
    linkNodeHelper(x, y+1, numRows, numCols, input_map, new_level, node, 'right')
    linkNodeHelper(x+1,  y, numRows, numCols, input_map, new_level,node, 'bottom')
    linkNodeHelper(x, y-1, numRows, numCols, input_map, new_level,node, 'left')


def linkNodeHelper( x, y, numRows, numCols, input_map, new_level,node, attrname):
    '''
    Function checks:
     1. if (x,y) are within boundaries
     2. if (x,y) corresponds to an unexplored location (1) or the target (9)
     3. creates the side node
     4. links the current node to the side node
    :param x: int representing x coordinate to check
    :param y: int representing t coordinate to check
    :param numRows: int of # of rows
    :param numCols: int of # of cols
    :param input_map: 2D array containing 1s, -1s, 0s and the target 9.
    :param new_level: Level of the side node
    :param node: current node for which we are linking the children
    :param attrname: string representing the 'side' attribute e.g. 'left'

    '''

    if 0 <= x and x < numRows \
            and 0 <= y and y < numCols \
            and (input_map[x][y] == 1 or input_map[x][y] == 9):
        # Create side node
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

def paint_path(node,img, img_artist, fg, sleep_rate = 0.1):

    if node:

        x,y = node.coordinates
        img[x][y] = [122/250,130/250,239/250] # dark grey
        img_artist.set_data(img)
        fg.canvas.draw()
        fg.canvas.flush_events()
        time.sleep(sleep_rate)

        paint_path(node.parent,img, img_artist, fg)


def apply_BFS(numRows, numCols, input_map, sleep_rate = 0.1):

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
    # Image settings
    matplotlib.use('TkAgg')
    fg = figure()
    axs = plt.gca()
    axs.set_xticks(range(numCols))
    axs.set_yticks(range(numRows))
    axs.xaxis.tick_top()

    img_artist = axs.imshow(img)
    plt.show(block=False)
    plt.title('Applying BFS on a '+str(numRows)+'x'+str(numCols)+' Matrix')

    plt.pause(1) # pause() takes too much time within loop

    while queue.queue:
        # Get node
        node = queue.dequeue()

        if node.data == 9:
            plt.xlabel('Distance to Target: '+ str(node.level))
            paint_path(node,img, img_artist, fg, sleep_rate)
            return node.level

        # link current node to  children not yet explored
        linkNodeToChildren(node, numRows, numCols, input_map) # if not node.linked: l

        # if children at top/right/bottom/left, then add to queue
        if node.top:
            queue.push(node.top)
        if node.right:
            queue.push(node.right)
        if node.bottom:
            queue.push(node.bottom)
        if node.left:
            queue.push(node.left)

        # Mark current node as visited
        x,y = node.coordinates
        input_map[x][y] = -1
        # Mark visited node on image
        img[x][y] = [234/250,242/250,128/250]
        img_artist.set_data(img)
        # fg.canvas.restore_region(background)
        #axs.draw_artist(img_artist)
        axs.set_xlabel('Distance to Target: '+str(node.level))
        # fg.canvas.blit(axs.bbox)
        fg.canvas.draw()

        #plt.pause(0.1)
        time.sleep(sleep_rate)

    # plt.close(fg)
    # plt.clf()
    # fg.clf()
    return -1


def get_distance(numRows, numCols, input_map, sleep_time = 0.1):
    '''

    :param numRows: int of # of rows
    :param numCols: int of # of cols
    :param input_map: 2D array containing 1s, -1s, 0s and the target 9.
    :return: minimum distance to get from top left (origin) to target 9, otherwise returns -1
    '''
    min_distance = apply_BFS(numRows, numCols, input_map, sleep_time)
    return min_distance

def get_map(m,n):
    ''' builds random map of mxn with at least one route from origin to target'''
    X = np.random.rand(m,n)
    X[X>=0.5] = 1
    X[X< 0.5] = 0
    X[0, :] = np.ones(n)
    X[:,-1] = np.ones(m)
    X[-1,-1] = 9
    return X

if __name__ == '__main__':

    # 3x3 case
    numRows = 3
    numCols = 3
    input_map = [[1,0,9], [1,0,1], [1,1,1]]
    print(get_distance(numRows, numCols, input_map, 0.6))

   #  5x5 case
    numRows = 5
    numCols = 5
    input_map = [[1,1,1,1,1],
                 [1,0,0,0,1],
                 [1,0,9,1,1],
                 [1,0,1,0,1],
                 [1,1,1,1,1]]
    print(get_distance(numRows, numCols, input_map, 0.3))

    # 8x8 case
    numRows = 8
    numCols = 8
    input_map = [[1, 1, 1, 1, 1, 1, 0, 0],
                 [1, 0, 0, 0, 0, 1, 0, 0],
                 [1, 0, 0, 1, 1, 1, 1, 1],
                 [1, 0, 0, 1, 0, 0, 0, 1],
                 [1, 0, 0, 1, 0, 0, 1, 1],
                 [1, 1, 1, 1, 0, 1, 1, 0],
                 [1, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 1, 1, 0, 0, 9]]
    print(get_distance(numRows, numCols, input_map, 0.1))


    # mxn case
    m = 20
    n = 20
    X = get_map(m,n)
    print(get_distance(m, n, X, 0.003))
    plt.show()
