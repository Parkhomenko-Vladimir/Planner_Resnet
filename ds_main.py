from ds_class import Dstar
import numpy as np

d_goal = [0,0]
d_start = [10,10]

dworld = np.zeros((50,50))

d_star = Dstar(dworld, d_goal, d_start, dworld)
d_star.Move_plan()
d_star_path = d_star.query(d_start)
print(d_star_path)