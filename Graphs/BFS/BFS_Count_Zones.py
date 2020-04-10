import matplotlib.pyplot as plt
from  matplotlib.pyplot import figure, grid
import matplotlib
import numpy as np
from utils.util import Node, PriorityQueue
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
            and input_map[x][y] != -1:
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


def get_zones(numRows, numCols, input_map, sleep_rate = 0.1):

    # Initialize root node
    data = input_map[0][0]
    root = Node([0,0], data)
    root.level = 0

    # Initialize queue
    p_queue = PriorityQueue()
    p_queue.push(root, data)

    # Initialize image
    img = np.zeros((numRows,numCols,3))
    img = create_img(img, numRows, numCols, input_map)
    # Image settings
    matplotlib.use('TkAgg')
    fg = figure()
    axs = plt.gca()
    # axs.set(xlim=(-0.5, numCols-0.5), ylim=(-0.5, numRows-0.5))
    axs.set_xticks(range(numCols))
    axs.set_yticks(range(numRows))

    # xdata_grid = np.arange(0.5,numCols)
    # ydata_grid = np.arange(0.5,numRows)
    # grid(color='y', linestyle='-') #, xdata = xdata_grid, ydata=ydata_grid)
    # #axs.axhline([x for x in xdata_grid], linestyle='--', color='k') # horizontal lines
    #
    # axs.set_yticks(xdata_grid, minor=True)
    # axs.set_xticks(xdata_grid, minor=True)


    img_artist = axs.imshow(img)
    plt.show(block=False)
    plt.title('Applying BFS on a '+str(numRows)+'x'+str(numCols)+' Matrix')


    plt.pause(1) # pause() takes too much time within loop

    # Zone flag
    prev_node_in_zone = False
    num_zones = 0
    visited_number_of_nodes = 0

    # Hash table for coordinates that has been in queue
    coordinates_dict = {}
    while p_queue.getQueue():
        # print(p_queue.getQueue())

        # # Check if we have visited all priority nodes in teh graph
        if visited_number_of_nodes + len(p_queue.queue) >= numRows*numCols:
            # print("Done searching: \nQueue is: ")
            # print(p_queue.getQueue())
            # print("visited_number_of_nodes "+str(visited_number_of_nodes))
            break

        # Get node
        node = p_queue.dequeue()

        # count zones if coming from a 0 zone to a 1 zone
        if not prev_node_in_zone and node.data == 1: # if entering a zone
            num_zones += 1

        if node.data == 1:
            prev_node_in_zone = True
        else:
            prev_node_in_zone = False

        # link current node to  children not yet explored
        linkNodeToChildren(node, numRows, numCols, input_map)

        # if children at top/right/bottom/left and it has not been in the queue, then add to queue
        if node.top and str(node.top.coordinates) not in coordinates_dict:
            coordinates_dict[str(node.top.coordinates)] = True
            p_queue.push(node.top, node.top.data)
        if node.right and str(node.right.coordinates) not in coordinates_dict:
            coordinates_dict[str(node.right.coordinates)] = True
            p_queue.push(node.right, node.right.data)
        if node.bottom and str(node.bottom.coordinates) not in coordinates_dict:
            coordinates_dict[str(node.bottom.coordinates)] = True
            p_queue.push(node.bottom, node.bottom.data)
        if node.left and str(node.left.coordinates) not in coordinates_dict:
            coordinates_dict[str(node.left.coordinates)] = True
            p_queue.push(node.left, node.left.data)

        # Mark zone node
        x,y = node.coordinates
        input_map[x][y] = -1 # visited
        visited_number_of_nodes += 1
        if node.data == 1:
            # Mark visited node on image corresponding to a zone
            img[x][y] = [234/250,242/250,128/250]
            img_artist.set_data(img)
        else:
            # Mark visited node on image corresponding to a point out of the zones
            img[x][y] = [100/250,100/250,100/250]
            img_artist.set_data(img)
        # fg.canvas.restore_region(background)
        axs.draw_artist(img_artist)
        fg.canvas.blit(axs.bbox)

        #plt.pause(0.1)
        time.sleep(sleep_rate)
    plt.xlabel('Number of Zones: '+ str(num_zones))
    # plt.close(fg)
    # plt.clf()
    # fg.clf()
    return num_zones

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

    # 3x3 case  #2 zones
    # numRows = 3
    # numCols = 3
    # input_map = [[1,0,0], [1,0,1], [1,0,1]]
    # print(get_zones(numRows, numCols, input_map, 1))

    # # 5x5 case   # 4
    # numRows = 5
    # numCols = 5
    # input_map = [[1,0,0,1,1],
    #              [1,0,0,1,0],
    #              [0,1,1,0,0],
    #              [0,0,0,0,0],
    #              [1,1,1,1,0]]
    # print(get_zones(numRows, numCols, input_map, 0.3))

    # # 8x8 case
    numRows = 8 #int(input())
    numCols = 8 #int(input())
    input_map = [[1, 1, 1, 1, 1, 1, 0, 0],
                 [1, 1, 0, 1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 1, 1, 1, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 1, 0],
                 [1, 1, 0, 0, 1, 1, 1, 1],
                 [1, 1, 1, 1, 0, 1, 1, 1]]
    print(get_zones(numRows, numCols, input_map, 0.1))
    #
    #
    # # mxn case
    # m = 20
    # n = 20
    # X = get_map(m,n)
    # print(get_distance(m, n, X, 0.005))
    plt.show()
