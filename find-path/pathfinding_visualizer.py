import pygame
import random
import sys
from os import getcwd
from visualizer import Visualizer
from pygame.constants import K_1, K_2, K_3, K_4
from data_structures import Queue, Stack, PriorityQueue
from typing import List



########### Global constants ##########
SIZE = 640                                  # size of the grid displayed on screen
BLOCKS_EACH_LINE = 80                       # number of blocks on each line
BLOCK_WIDTH = SIZE // BLOCKS_EACH_LINE      # width of each block
HALF_WIDTH = BLOCK_WIDTH // 2               # 1/2 width of each block
WIN_W = SIZE + 330                          # width of the screen (extra 330 for text)
WIN_H = SIZE                                # height of the screen

START_POS = (6, 6)                                      # index for start block
END_POS = (BLOCKS_EACH_LINE - 7, BLOCKS_EACH_LINE - 7)  # index for end block

# colors
BLACK = (0, 0, 0)
YELLOW = (244, 242, 140)
GREEN = (10, 225, 20)
WHITE = (255, 255, 255)

_looping = True     # keep the mainloop run




#################################################################################
#######################  PATHFINDING VISUALIZER CLASS  ##########################
# Create the GUI and visualize the pathfinding process for different algorithms #
#-------------------------------------------------------------------------------#

class PathfindingVisualizer(Visualizer):

    def __init__(self) -> None:
        ##### initialize variables #####
        global _looping
        self._grid = []                  # the grid containing all the blocks 
        self._cleared = True             # if the grid is cleared (all walkable)
        _looping = True                  # keep the mainloop running
        self._generated = False          # if random barriers are already generated

        ##### initialize the screen display #####
        icon_path = getcwd() + "/images/path_icon.ico"
        super().__init__(WIN_W, WIN_H, 'Pathfinding Visualizer', WHITE, icon_path)
        self.__show_instruction_text()    # show the instruction text
        self.__create_blocks()            # create the blocks for the grid
        self.__mainloop()




    def __mainloop(self):
        while _looping:
            super().draw() 
            self.__input_handling()
                       



    def __input_handling(self) -> None:
        """
        Handle mouse and keyboard input from user
        """
        # if the left mouse is pressed -> draw barrier
        if pygame.mouse.get_pressed() == (1,0,0):
            pos = pygame.mouse.get_pos()
            self.__update_block_clicked(pos, "barrier")
            self._cleared = False
        # if the right mouse is pressed -> delete
        elif pygame.mouse.get_pressed() == (0,0,1):
            pos = pygame.mouse.get_pos()
            self.__update_block_clicked(pos, "walkable")

        for event in pygame.event.get():
            # click exit -> quit and return to menu
            if event.type == pygame.QUIT:
                quit()

            # If there is a Key pressed 
            elif event.type == pygame.KEYDOWN:

                # Esc Key -> quit and return to menu
                if event.key == pygame.K_ESCAPE:
                    quit()

                # C -> clear the grid
                elif event.key == pygame.K_c and not self._cleared:
                    self._generated = False
                    self._cleared = True
                    self.__clear()

                # G -> randomly generate obstacles
                elif event.key == pygame.K_r and not self._generated:
                    self._generated = True
                    self._cleared = False
                    self.__generate_obstacles()

                # Return Key -> Start
                elif event.key == pygame.K_RETURN:
                    self._generated = True
                    # check all the neighbors before starting the algorithm
                    self.__check_all_neighbors()
                    self.__start_finding()

                # number -> choose the corresponding algorithm
                else:
                    switch = {
                        K_1: 1,
                        K_2: 2,
                        K_3: 3,
                        K_4: 4
                    }
                    chosen = switch.get(event.key, -1)
                    self.__pick_algo(chosen)
 
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # Ctrl + right click -> set start point
                    if event.button == 1:
                        self.__update_block_clicked(pos, "start")
                    # Ctrl + left click -> set end point
                    else:
                        self.__update_block_clicked(pos, "end")




    def __create_blocks(self) -> None:
        """
        create and add the blocks on the grid
        """
        # add new blocks
        for i in range(BLOCKS_EACH_LINE):
            self._grid.append([])
            for j in range(BLOCKS_EACH_LINE):
                # when the display is not fully initialized
                # no need to make expanding effect
                # so we set the effect to be False
                new_block = Block(self._screen, i, j, False)
                # add new block to the grid
                self._grid[i].append(new_block) 
        # iniitalize the start and end blocks
        self.__init_start_end_points()              




    def __show_instruction_text(self) -> None:
        """
        Display the instruction text on the right side of the grid
        """
        pos_y = 100
        self.font = pygame.font.Font('JhengHei.ttf', 20, bold=True)
        self.font1 = pygame.font.Font('JhengHei.ttf', 30, bold=True)
        
        # the list of instructions to display
        instruction_list = [
            "滑鼠左鍵: 建立路障",
            "滑鼠右鍵: 移除路障",
            "Ctrl + 滑鼠左鍵: 設定起始位置",
            "Ctrl + 滑鼠右鍵: 設定終點位置",
            "R： 隨機建立路障",
            "C： 重製地圖",
            "<Enter>： 開始搜尋",
            "<ESC>： 結束",
            "使用的演算法 Algorithm:",]
        
        # show all the instructions on the screen
        for idx, string in enumerate(instruction_list):
            text = self.font.render(string, True, BLACK)
            self._screen.blit(text, (SIZE + 10, pos_y))
            if idx == len(instruction_list) - 2:
                pos_y += 160
            else:
                pos_y += 35
            
        ### display the list of algorithms to choose below the instruction text
        
        
        pygame.draw.line(self._screen, BLACK, (SIZE, 0), (SIZE, SIZE), 4)
        pygame.draw.rect(self._screen, BLACK, [SIZE, 20, 200, 60], width=3, border_bottom_right_radius=10, border_top_right_radius=10)
        self._control = self.font1.render('操作方式', True, BLACK)
        self._screen.blit(self._control , ((SIZE + ((200 - self._control.get_rect()[2]) // 2)), (20 + ((60 - self._control.get_rect()[3]) // 2))))
        
        y_height = 390
        icon_name = [('icon1.png', '起點'), ('icon2.png', '終點'), ('icon3.png', '障礙物')]
        for path in icon_name:
            icon = pygame.image.load(path[0])
            icon = pygame.transform.scale(icon, (20, 20))
            icon.convert()
            self._screen.blit(icon, (SIZE + 20, y_height))
            self._iconl_name = self.font.render(path[1], True, BLACK)
            self._screen.blit(self._iconl_name , ((SIZE + 20 + 35), y_height - 5))
            y_height += 35
            
        
        
        # list of algorithms' names
        self.algo_names = [
            " Dijkstra Algorithm",
            "   ",
            "  ",
            "  "]
        
        self.algo_text = []     # rendered text for the algorithms names
        self.algo_pos_y = []    # y position for each algorithm text
        
        for i in range (0, len(self.algo_names)):
            self.algo_pos_y.append(pos_y)
            self.algo_text.append(self.font.render(
                self.algo_names[i], True, BLACK))
            self._screen.blit(self.algo_text[i], (SIZE+10, self.algo_pos_y[i]))
            pos_y += 25
            
        # default algorithm is Dijkstra
        self.algo_picked = 2
        self.__pick_algo(2)




    def __pick_algo(self, n: int) -> None:
        """
        Action made when the user chooses an algorithm to visualize
        Args:
            n (int): the number associated to the algorithm
        """
        if n != -1:
            # reset the previously chosen algorithm text to green
            self.algo_text[self.algo_picked-1] = self.font.render(
                self.algo_names[self.algo_picked-1], False, GREEN)
            self._screen.blit(self.algo_text[self.algo_picked-1], 
                (SIZE+10, self.algo_pos_y[self.algo_picked-1]))

            # set the newly chosen algorithm text to yellow
            self.algo_text[n-1] = self.font.render(
                self.algo_names[n - 1], False, YELLOW)
            self._screen.blit(self.algo_text[n-1], 
                (SIZE+10, self.algo_pos_y[n-1]))

            # set the chosen algorithm
            self.algo_picked = 2




    def __update_block_clicked(self, pos: tuple, status: str) -> None:
        """
        Get the block that was clicked and change its status
        Args:
            pos (tuple): position of the block
            status (str): the status to be changed to
        """
        x, y = pos
        if x < SIZE and y < SIZE:
            # get the index of the block clicked
            x = x // BLOCK_WIDTH
            y = y // BLOCK_WIDTH

            if ((x, y) != self.start_point and (x, y) != self.end_point):
                if status == "walkable":
                    self._grid[x][y].set_walkable()
                elif status == "barrier":
                    self._grid[x][y].set_barrier()
                elif status == "start":
                    if self.start_point != None:
                        i, j = self.start_point
                        self._grid[i][j].set_walkable()
                    self.start_point = (x, y)
                    self._grid[x][y].set_start()
                else:
                    if self.end_point != None:
                        i, j = self.end_point
                        self._grid[i][j].set_walkable()
                    self.end_point = (x, y)
                    self._grid[x][y].set_end()




    def __clear(self) -> None:
        """
        Reset the grids to its initial state
        """
        # reset all the blocks on the screen
        for i in range(BLOCKS_EACH_LINE):
            for j in range(BLOCKS_EACH_LINE):
                self._grid[i][j].reset()
        # put the start and end blocks to default locations
        self.__init_start_end_points()




    def __generate_obstacles(self) -> None:
        """
        Randomly generate the obstacles on the grid
        """
        # generate the obstacles randomly
        for _ in range(SIZE * 4 // 3):
            rand_x = random.randint(0, BLOCKS_EACH_LINE - 1)
            rand_y = random.randint(0, BLOCKS_EACH_LINE - 1)
            if self._grid[rand_x][rand_y].is_walkable():
                self._grid[rand_x][rand_y].set_barrier()
        sx, sy = self.start_point
        ex, ey = self.end_point

        # clear the blocks around the start and end blocks so the they won't be covered
        for i in range (-4, 4):
            for j in range (-4, 4):
                # make sure the index that we are going to clear is in the valid range
                if sx+i >= 0 and sx+i <= BLOCKS_EACH_LINE-1:
                    if sy+j >= 0 and sy+j <= BLOCKS_EACH_LINE-1:
                        if not self._grid[sx+i][sy+j].is_start_block():
                            if not self._grid[sx+i][sy+j].is_end_block():
                                self._grid[sx+i][sy+j].set_walkable()
                if ex+i >= 0 and ex+i <= BLOCKS_EACH_LINE-1:
                    if ey+j >= 0 and ey+j <= BLOCKS_EACH_LINE-1:
                        if not self._grid[ex+i][ey+j].is_start_block():
                            if not self._grid[ex+i][ey+j].is_end_block():
                                self._grid[ex+i][ey+j].set_walkable()
    



    def __init_start_end_points(self) -> None:
        """
        Put the start and end blocks to their default locations
        """
        self.start_point = START_POS
        self.end_point = END_POS
        self._grid[START_POS[0]][START_POS[1]].set_start()
        self._grid[END_POS[0]][END_POS[1]].set_end()
        



    def __check_all_neighbors(self) -> None:
        """
        Update the neighbors for each block on the grid
        """
        for row in self._grid:
            for block in row:
                block.update_neighbors(self._grid)

    


    def __start_finding(self) -> None:
        """
        Start the pathfinind process
        """
        self._cleared = False
        start_block = self._grid[self.start_point[0]][self.start_point[1]]
        switch = {
            1: lambda grid, start_block, end_pos: a_star(grid, start_block, end_pos),
            2: lambda grid, start_block, end_pos: dijkstra(grid, start_block),
            3: lambda grid, start_block, end_pos: breadth_first(start_block),
            4: lambda grid, start_block, end_pos: depth_first(start_block)
        }
        switch.get(self.algo_picked)(self._grid, start_block, self.end_point)





########################################################################
########################### BLOCK CLASS ################################
# Represent a "pixel" on the map. Many of these blocks will make up our 
# map. Each block will have its own status such as visited, walkable, 
# barrier, etc. 
#----------------------------------------------------------------------#

class Block():

    # we will use color to determine the status of the block
    PATH = LIGHT_BLUE = (89, 205, 225)          # is part of the shortest path
    START = PINK = (49, 58, 181)              # is the start point
    END = RED = (215, 17, 27)                   # is the end point
    BARRIER = DARK_BLUE = (0, 0, 0)         # is walkable
    WALKABLE = WHITE = (220, 220, 220)          # is a barrier       
    VISITED = LIGHT_YELLOW = (255, 205, 102)    # is already visited
    NEXT_TO_VISIT = GREEN = (195, 255, 105)     # is in the waitlist to be visited


    def __init__(self, screen, x, y, effect=True) -> None:
        """
        Args:
            screen ([type]): the screen where the block will be placed on
            x ([type]): the x-coordinate of the block
            y ([type]): the y-coordinate of the block
            effect (bool, optional): true to make the expanding effect (defaults to True)
        """
        self._screen = screen        # root screen to dislay the block
        self.x, self.y = x, y       # row and column index where the block is at
        self.neighbors = []         # the list of neighbors
        self.parent = None          # the parent of the block
        self.was_visited = False    # if the block is visited or not

        ### The actual position on the map (since each block has a width) ###
        self.pos_x = (x * BLOCK_WIDTH) + HALF_WIDTH
        self.pos_y = (y * BLOCK_WIDTH) + HALF_WIDTH

        ### display the block ###
        self.image = pygame.Surface([BLOCK_WIDTH, BLOCK_WIDTH]) # the size of the block
        self.rect = self.image.get_rect()                       # generate the block as a rectangle
        self.rect.center = [self.pos_x, self.pos_y]             # put the rectangle in its position
        self.__update_status(self.WALKABLE, effect)             # initial status is walkable


    def is_walkable(self) -> bool:
        return self.color == self.WALKABLE

    def is_barrier(self) -> bool:
        return self.color == self.BARRIER

    def is_next(self) -> bool:
        return self.color == self.NEXT_TO_VISIT

    def is_visited(self) -> bool:
        return self.color == self.VISITED or self.was_visited

    def is_start_block(self) -> bool:
        return self.color == self.START

    def is_end_block(self) -> bool:
        return self.color == self.END

    def get_position(self) -> tuple:
        return self.x, self.y

    def get_color(self) -> tuple:
        return self.color

    def get_neighbors(self) -> list:
        return self.neighbors

    def get_parent(self):
        return self.parent


    def __update_status(self, color_status, effect=True) -> None:
        """
        Update the color status for the block
        Args:
            color_status ([type]): the new color of the block
            effect (bool, optional): true to make the expanding effect (defaults to True)
        """
        self.color = color_status                                   # set the color_status
        pygame.draw.rect(self._screen, color_status, self.rect)     # draw block with new color

        # make the expanding effect
        if effect:       
            pygame.display.update()
        if color_status is self.PATH:
            pygame.time.delay(5)        # if a block is path, slow the expanding effect down 


    def set_path(self) -> None:
        """
        Set the status of the current block to be a part of the path
        """
        self.__update_status(self.PATH)


    def set_barrier(self) -> None:
        """
        Set the status of the current block to be an obstacle/barrier
        """
        self.__update_status(self.BARRIER)


    def set_walkable(self) -> None:
        """
        Set the current block's status to be walkable
        """
        self.__update_status(self.WALKABLE)
        

    def set_start(self) -> None:
        """
        Set the current block to be the start block
        """
        self.__update_status(self.START)


    def set_end(self) -> None:
        """
        Set the current block to be the end block
        """
        self.__update_status(self.END)


    def set_next(self) -> None:
        """
        Set the status to be the next block
        """
        if self.is_walkable():
            self.__update_status(self.NEXT_TO_VISIT)


    def set_parent(self, parent) -> None:
        """
        Set the parent block for the current block
        Args:
            parent ([type]): the parent block
        """
        self.parent = parent


    def set_visited(self) -> None:
        """
        Set the status to be visited
        """
        self.was_visited = True
        if self.is_walkable() or self.is_next():
            self.__update_status(self.VISITED)


    def reset(self) -> None:
        """
        Reset the block
        """
        self.__update_status(self.WALKABLE, False)
        self.was_visited = False
        self.neighbors = []
        self.parent = None


    def update_neighbors(self, grid) -> None:
        """
        Check the 4 directions (N, E, S, W) around the block and add the neighbor
        block to this block's list of neighbors if qualified
        Args:
            grid ([type]): the grid containing all blocks
        """
        # make sure the index does not go out of the array's range
        north_has_block = self.x > 0
        south_has_block = self.x < BLOCKS_EACH_LINE - 1
        west_has_block = self.y > 0
        east_has_block = self.y < BLOCKS_EACH_LINE - 1

        # check the neighbors in clock-wise order
        # if the block is not a barrier, then add it to the neighbor list
        # check north
        if north_has_block:
            if not grid[self.x - 1][self.y].is_barrier():
                self.neighbors.append(grid[self.x-1][self.y])
        # check east:
        if east_has_block:
            if not grid[self.x][self.y + 1].is_barrier():
                 self.neighbors.append(grid[self.x][self.y+1])
        # check south
        if south_has_block:
            if not grid[self.x + 1][self.y].is_barrier():
                self.neighbors.append(grid[self.x+1][self.y])
        # check west:
        if west_has_block:
            if not grid[self.x][self.y - 1].is_barrier():
                self.neighbors.append(grid[self.x][self.y-1])        
            





#######################################################################################
####################  P A T H F I N D I N G   A L G O R I T H M S  ####################
#######################################################################################




############################
#### DEPTH FIRST SEARCH ####

def depth_first(start_block: Block) -> None:
    """
    Depth first search algorithm
    Args:
        start_block (Block): the start block
    """
    found = False
    stack = Stack()
    # initialize the stack with the stack block
    stack.push(start_block)

    while not stack.is_empty() and _looping:
        current: Block = stack.pop()
        if current.is_end_block():
            found = True
            break
        for neighbor in current.get_neighbors():
            if not neighbor.is_visited() and not neighbor.is_next():
                neighbor.set_next()
                neighbor.set_parent(current)
                stack.push(neighbor)
        current.set_visited()
        __input_handling()
    if _looping:
        if found:
            # backtrack to show the path
            __backtrack(current.get_parent())




##############################
#### BREADTH FIRST SEARCH ####

def breadth_first(start_block: Block) -> None:
    """
    Breadth first search algorithm
    Args:
        start_block (Block): the start block
    """
    found = False
    queue = Queue()
    # initialize the queue with the start block
    queue.enqueue(start_block)
    
    while not queue.is_empty() and _looping:
        current: Block = queue.dequeue()
        if current.is_end_block():
            found = True
            break
        for neighbor in current.get_neighbors():
            if not neighbor.is_visited() and not neighbor.is_next():
                neighbor.set_parent(current)
                neighbor.set_next()
                queue.enqueue(neighbor)
        current.set_visited()
        __input_handling()
    if _looping:
        if found:
            # backtrack to show the path
            __backtrack(current.get_parent())

############################
#### DIJKSTRA ALGORITHM ####

def dijkstra(grid, start_block: Block) -> None:
    """
    Dijkstra algorithm
    Args:
        grid ([type]): the grid containing all blocks
        start_block (Block): the start block
    """

    # 定義有沒有找到終點
    found = False
    # 建立一個優先序列
    prio_queue = PriorityQueue()
    # 使用字典生成式將二維串列攤平並賦予無限大的值
    dis = {block: sys.maxsize for row in grid for block in row}
    # 將起點的值預設為 0
    dis[start_block] = 0

    # 將起點排入佇列當中
    prio_queue.enqueue(start_block, dis[start_block])

    # 在佇列不為空，或 pygame 迴圈沒有停止時不斷執行
    while not prio_queue.is_empty() and _looping:
        # 將佇列的首端(head)做 pop 成為 current 當前節點
        current: Block = prio_queue.dequeue()
        # 找到終點，break
        if current.is_end_block():
            # 找到終點
            found = True
            break
        # 尋找當前節點(current)的鄰居(上下左右的節點)
        for neighbor in current.get_neighbors():
            # 確認此節點有沒有被走訪過
            if not neighbor.is_visited() and not neighbor.is_next():
                # 暫存(temp)，因為上下左右的鄰居距離皆為 1，所以加 1 做判斷
                temp = dis[current] + 1
                # 如果此距離(dis)比原有鄰居的距離短，則將鄰居的 dis 取代為 temp
                # 並設下 set_parent 最後會使用遞迴的方式將最短路徑找出，並畫出
                if temp < dis[neighbor]:
                    dis[neighbor] = temp
                    neighbor.set_parent(current)
                    neighbor.set_next()
                    prio_queue.enqueue(neighbor, dis[neighbor])
        # 將當前節點設為已走訪
        current.set_visited()
        # 按鈕控制函式
        __input_handling()

    if _looping:
        
        # 找到終點後往回顯示最短路徑
        if found:
            __backtrack(current.get_parent())




######################
#### A* ALGORITHM ####

def a_star(grid, start_block: Block, end_pos: tuple) -> None:
    """
    A* algorithm
    Args:
        grid ([type]): the grid containing all blocks
        start_block (Block): the starting block
        end_pos (tuple): the position of the end block
    """
    found = False
    p_queue = PriorityQueue()
    # for each block on the grid
    # g cost = distance to the starting block
    # h cost (heuristic) = distance to the end block
    # f cost (which is our priority) = g cost + h cost
    g_cost = {block: sys.maxsize
              for row in grid for block in row}
    g_cost[start_block] = 0
    h_cost = {block: __get_heuristic(block, end_pos) 
              for row in grid for block in row}
    f_cost = {block: (g_cost[block] + h_cost[block]) 
              for row in grid for block in row}

    # initalize the queue with the start block
    p_queue.enqueue(start_block, f_cost[start_block])

    # keep searching until the queue is empty
    while not p_queue.is_empty() and _looping:
        current: Block = p_queue.dequeue()
        # if the end block is found then stop
        if current.is_end_block():
            found = True
            break
        for neighbor in current.get_neighbors():
            if not neighbor.is_visited() and not neighbor.is_next():
                # since each block is next to each other,
                # the g cost of neighbor is simply 1 more than that of current
                g_temp = g_cost[current] + 1
                # since h cost is constant
                # f cost only varies based on g cost
                if g_temp < g_cost[neighbor]:
                    g_cost[neighbor] = g_temp
                    f_cost[neighbor] = g_temp + h_cost[neighbor]
                    neighbor.set_parent(current)
                    neighbor.set_next()
                    p_queue.enqueue(neighbor, f_cost[neighbor])
        current.set_visited()
        
        
        
        __input_handling()

    if _looping:
        if found:
            __backtrack(current.get_parent())




def __get_heuristic(block: Block, end_pos: tuple) -> float:
    """
    Calculate the heuristic distance for a block
    """
    block_pos = block.get_position()
    x = abs(block_pos[0] - end_pos[0])
    y = abs(block_pos[1] - end_pos[1])
    return x + y

##############################################################################



def __backtrack(root: Block) -> None:
    """
    Backtrack to highlight the shortest path
    Args:
        root (Block): the previous block visitted
    """
    while not root.is_start_block():
        root.set_path()             # set the block to be part of the path
        root = root.get_parent()    # keep backtracking




def __input_handling() -> None:
    """
    Handle mouse and keyboard input when the pathfininding process is running
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
        elif event.type == pygame.KEYDOWN:
            # Esc Key -> quit and return to menu
            if event.key == pygame.K_ESCAPE:
                quit()




def quit() -> None:
    """
    Close the pathfininding visualizer and re-open the home screen
    """
    global _looping
    _looping = False
    pygame.display.quit()




''' end of pathfinding_visualizer.py'''