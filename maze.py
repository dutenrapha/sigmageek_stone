import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from scipy.ndimage import convolve
from operator import itemgetter

class Maze:
    def __init__(self, initial_position, goal, current_state) ->  None:
        """Initialize the Maze class parameters

        Parameters
        ----------
        initial_position : _type_
            tuple with initial position
        goal : _type_
            tuple with goal position
        current_state : _type_
            current Maze state matrix
        """      
        self.initial_position = initial_position
        self.goal = goal
        self.current_state = current_state
        self.update_num = 0
        self.rows = len(self.current_state)
        self.columns = len(self.current_state[0])
        self.response = None
        self.path = None

    def _update_rule(self, temp:np.array, M_sum:np.array, i:int, j:int) ->  None:
        """_summary_

        Parameters
        ----------
        temp : np.array
            Temporary new maze state
        M_sum : np.array
            Neighbors sum matrix for the current maze state
        i : int
            x coordinate to be updated
        j : int
            y coordinate to be updated
        """        
        if (M_sum[i][j] > 1 and M_sum[i][j] < 5) and self.current_state[i][j] == 0:
            temp[i][j] = 1
        if (M_sum[i][j] > 3 and M_sum[i][j] < 6) and self.current_state[i][j] == 1:
            temp[i][j] = 1

    def _update(self) -> None:
        """Update the current state of the maze acording to the _update_rule
        """        
        temp = np.zeros((self.rows,self.columns))
        kernel = [[1, 1, 1],[1, 0, 1],[1, 1, 1]]
        M_sum = convolve(self.current_state, kernel, mode='constant')
        for i in range(self.rows):
            for j in range(self.columns):
                self._update_rule(temp,M_sum,i,j)
        self.current_state = temp
        self.current_state[self.rows - 1][self.columns - 1] = 0
        self.current_state[0][0] = 0
        self.update_num += 1

    def _next_path(self,point:tuple) -> List[Tuple]:
        """Given a tuple with the current position create a list of tuples with the next possible moves

        Parameters
        ----------
        point : tuple
            current position coordinate

        Returns
        -------
        List[Tuple]
            list of all possible moves
      """         
        temp = []
        if point[0] - 1 >= 0:
            if self.current_state[point[0] - 1][point[1]] != 1:
                temp.append((point[0] - 1,point[1]))
        if point[0] + 1 < self.rows:  
            if self.current_state[point[0] + 1][point[1]] != 1:
                temp.append((point[0] + 1,point[1]))
        if point[1] - 1 >= 0:
            if self.current_state[point[0]][point[1] - 1] != 1:
                temp.append((point[0],point[1] - 1))
        if point[1] + 1 < self.columns:
            if self.current_state[point[0]][point[1] + 1] != 1:
                temp.append((point[0],point[1] + 1))
        return(temp)

    def _is_goal(self) -> None:
        """Check if the was reached in the current turn
        """
        chave = 0
        for kk, vv in self.response.items():
            for k, _ in vv:
                if k == self.goal:
                    chave = 1
                    break
            if chave == 1:
                break
        if chave == 0:
            return(False)
        else:
            return(True)

    def _manhattan(self, a, b):
        return sum(abs(val1-val2) for val1, val2 in zip(a,b))

    def _check_distance(self) -> tuple:
        """Check if the was reached in the current turn
        """
        pair_min = (0,0)
        d_min = 1000000
        for _, vv in self.response.items():
            temp =  [i[0] for i in vv]
            if len(temp) > 10:
                a = max(temp,key=itemgetter(0))
                b = max(temp,key=itemgetter(1))
                d_tmep = self._manhattan(a, self.goal)
                if d_tmep < d_min:
                    d_min = d_tmep
                    pair_min = a
                d_tmep = self._manhattan(b, self.goal)
                if d_tmep < d_min:
                    d_min = d_tmep
                    pair_min = b
        return(pair_min)

    def _create_maze(self)->None:
        """Populates the self.response dict whose key is the current turn and the values are a list of tuples. Each tuple shows the possible current positions, in its first position, and the possible next moves. 
        """      
        turns = {} 
        origin = [self.initial_position]
        self._update()
        i = 0
        start = time.time()
        global_start = time.time()        
        while True:
            turns[i] = []
            for p in origin:
                turns[i].append((p, self._next_path(p)))
            ter = set()    
            for _ , kk in turns[i]:
                for k in kk:
                    ter.add(k)
            origin = list(ter)
            self.response = turns
            self._update()
            i += 1
            if time.time() - start > 10:
                print(f"Greatest distance reached {self._check_distance()} in {time.time() - global_start}")
                start = time.time()
            if self._is_goal():
                break

    def _path_finder(self)->None:
        """Use the self.reponse data to generate a list of tuples with all points from origin to self.goal.
        """      
        self._create_maze()
        temp = list(reversed(self.response.values()))
        current = self.goal
        resp = [current]
        for p in temp[1:]:
            for k, v in p:
                if current in v:
                    current = k
                    resp.append(k)
        self.path = list(reversed(resp))

    def _postion(self,a:tuple,b:tuple)->str:
        """Given two points, a and b, it shows which direction should be taken to leave point a and arrive at point b.

        Parameters
        ----------
        a : tuple
            point with coordinate x in the first position and y in the second position.
        b : tuple
            point with coordinate x in the first position and y in the second position.

        Returns
        -------
        str
            One of the possible directions U for up, D for down, R for right and L for left
        """      
        x = a[0] - b[0]
        y = a[1] - b[1]
        if (x, y) == (1,0):
            return("U")
        if (x, y) == (-1,0):
            return("D")
        if (x, y) == (0,-1):
            return("R")
        if (x, y) == (0,1):
            return("L")
        
    def solve(self, filename:str = 'solution.txt') -> None:
        """Give a maze in filename print the soltuion and save it on a file called solution.txt

        Parameters
        ----------
        filename : str, optional
            filepath to a valid maze, by default 'solution.txt'
        """       
        self._path_finder()
        temp = []
        for i in range(1,len(self.path),1):
            temp.append(self._postion(self.path[i-1], self.path[i]))
        temp = ' '.join(temp)
        with open(filename, 'w') as f:
            f.write(temp)
        print(temp)
    
    def print_state(self):
        """Print the current state of the maze to the screen.
        """      
        plt.matshow(self.current_state, cmap=plt.cm.Blues)

if __name__ == '__main__':
    arg = sys.argv    
    if len(arg) == 2:
      M = np.loadtxt(arg[1])
      M[0][0] = 0
      M[M.shape[0] - 1][M.shape[1] - 1] = 0
      goal = M.shape[0] - 1, M.shape[1] - 1
      print(goal)
      maze = Maze((0,0),goal,M)
      maze.solve()
    else:
        print("Error: You need pass just ONE valid file path. E.g. python maze.py input.txt")
