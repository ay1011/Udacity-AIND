# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: Constraint propagation is the process of of finding a solution (i.e.search) that satisfies all the constraints,
or no solution is found that could satisfy all the constraints.

The naked twins is used to reduce the number of possibilities (branching factor) during the search.
A pair of BOXES (twins), belonging to the same set of PEERS, with the same 2 values as possibilities are identified.
These two values are then eliminated from all the BOXES that are PEERS to both the BOXES of the twins.

For example, if the naked twins are in A1 and A3, their common peers in the ROW A and SQUARE containing A1 and A3
would cannot assume the values in the naked twins. Thus, A2,B1,B2,B3,C1,C2,C3, A4,A5,A6,A7,A8,A9 will have the two values
contained in the naked twins reoved from them.


# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: A diagonal constraint is added as an additional UNIT in the diagonal sudoku problem.
The diagonal BOXES now have other diagonal BOXES as their peers
and are subject to the diagonal constraint, as well as, the constraints imposed by other UNITS (ROW, COLUMN, SQUARE).
The increased number of constraints reduces the potential values in a BOX.



### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solution.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in solution.py

### Submission
Before submitting your solution to a reviewer, you are required to submit your project to Udacity's Project Assistant, which will provide some initial feedback.  

The setup is simple.  If you have not installed the client tool already, then you may do so with the command `pip install udacity-pa`.  

To submit your code to the project assistant, run `udacity submit` from within the top-level directory of this project.  You will be prompted for a username and password.  If you login using google or facebook, visit [this link](https://project-assistant.udacity.com/auth_tokens/jwt_login for alternate login instructions.

This process will create a zipfile in your top-level directory named sudoku-<id>.zip.  This is the file that you should submit to the Udacity reviews system.

