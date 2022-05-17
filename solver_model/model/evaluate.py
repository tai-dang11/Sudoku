import numpy as np
from keras.models import load_model
import tensorflow as tf

# check if the predicted solution is correct
def valiadate_solution(puzzle: [[str]]):

    # check if row or column has correct order
    def valid(line):
        nums = [i for i in line]
        return sorted(nums) == [1,2,3,4,5,6,7,8,9]

    # check all row in puzzle
    def valid_row(puzzle):
        for row in puzzle:
            if not valid(row): return False
        return True

    # check all columns in puzzle
    def valid_column(puzzle):
        for column in puzzle.T:
            if not valid(column): return False
        return True

    # check 9 sub squares in puzzle
    def valid_square(puzzle):
        for i in range(0,len(puzzle),3):
            for j in range(0,len(puzzle),3):
                sub_square = [puzzle[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]
                if valid(sub_square) == False: return False
        return True

    if(valid_row(puzzle) and valid_square(puzzle) and valid_column(puzzle)):
        print("The solution is correct")
    else:
        print("Incorrect solution")


# predict the puzzle and check its validity
def sudoku_solver(puzzle, model_path, check = False):

    puzzle = np.array(puzzle)
    print("Original puzzle: ")
    print(puzzle)

    cnn_layer = load_model(model_path)

    model = tf.keras.Sequential([
        cnn_layer,
        tf.keras.layers.Softmax()
    ])
     # check if is there any 0 in the puzzle, if yes then change it
    while 0 in puzzle:

        normailzed_input = (puzzle / 9) - 0.5
        input = normailzed_input.reshape(1, 9, 9, 1)
        output = model.predict(input)

        out = np.argmax(output, axis=-1).squeeze() + 1
        max_out = np.max(output, axis=-1).squeeze()

        max_out[puzzle != 0] = -1.0

        index = np.where(max_out.max() == max_out)
        x, y = index

        indices = (x[0], y[0])
        puzzle[indices] = out[indices]

    print("Predicted puzzle:")
    print(puzzle)

    if check:
        print("\nCheck if the solution if valid:")
        valiadate_solution(puzzle)



puzzle = [[0,0,0,0,0,0,5,7,3],
         [8,0,0,0,2,0,0,0,0],
         [7,0,0,9,0,0,8,1,0],
         [5,8,0,7,0,6,0,0,0],
         [0,0,1,8,0,0,0,6,0],
         [2,3,0,0,4,0,0,0,9],
         [9,1,5,0,0,0,0,0,0],
         [0,0,0,0,8,0,6,0,1],
         [0,0,0,0,0,0,0,4,0]]

model_path = "../logs/4"
solution = sudoku_solver(puzzle, model_path, check = True)




