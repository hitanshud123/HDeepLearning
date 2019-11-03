import random
import math

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols   
        self.arr = []
        for i in range(rows):
            self.arr.append([])
            for j in range(cols):
                self.arr[i].append(0);
    
    @classmethod
    def from_array(cls, arr):
        n = cls(len(arr),1)
        for i in range(len(arr)):
            n.arr[i][0] = arr[i]
        return n
    
    @classmethod
    def from_array2d(cls, arr):
        n = cls(len(arr), len(arr[0]))
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                n.arr[i][j] = arr[i][j]
        return n

    @classmethod
    def copy(cls, a):
        b = Matrix(a.rows, a.cols)
        for i in range(a.rows):
            for j in range(a.cols):
                b.arr[i][j] = a.arr[i][j]
        return b
    
    # @classmethod
    # def from_dict(cls, matrix_dict):
    #     return cls()
    def to_array(self):
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.arr[i][j])
        return arr

    def to_array2d(self):
        arr = []
        for i in range(self.rows):
            arr.append([])
            for j in range(self.cols):
                arr[i].append(self.arr[i][j])
        return arr
    
    @property
    def shape(self):
        return (self.rows, self.cols)
        
    def reshape(self, rows, cols):
        m = Matrix(rows, cols)
        arr = self.to_array()
        for i in range(rows):
            for j in range(cols):
                m.arr[i][j] = arr[i*cols+j]
        return m

    def randomize(self, start=-1, end=1):
        for i in range(len(self.arr)):
            for j in range(len(self.arr[i])):
                self.arr[i][j] = random.random()*(end-start)+start

    def add(self, other):
        if (isinstance(other, Matrix)):
            for i in range(min(self.rows, other.rows)):
                for j in range(min(self.cols, other.cols)):
                    self.arr[i][j] += other.arr[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.arr[i][j] += other

    def sub(self, other):
        if (isinstance(other, Matrix)):
            for i in range(min(self.rows, other.rows)):
                for j in range(min(self.cols, other.cols)):
                    self.arr[i][j] -= other.arr[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.arr[i][j] -= other

    def mult(self, other):
        if (isinstance(other, Matrix)):
            for i in range(min(self.rows, other.rows)):
                for j in range(min(self.cols, other.cols)):
                    self.arr[i][j] *= other.arr[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.arr[i][j] *= other

    @staticmethod
    def s_add(a, b):
        m = Matrix(a.rows, a.cols)
        if (isinstance(b, Matrix)):
            for i in range(min(a.rows, b.rows)):
                for j in range(min(a.cols, b.cols)):
                    m.arr[i][j] = a.arr[i][j] + b.arr[i][j]
        else:
            for i in range(a.rows):
                for j in range(a.cols):
                    m.arr[i][j] = a.arr[i][j] + b
        return m

    @staticmethod
    def s_sub(a, b):
        m = Matrix(a.rows, a.cols)
        if (isinstance(b, Matrix)):
            for i in range(min(a.rows, b.rows)):
                for j in range(min(a.cols, b.cols)):
                    m.arr[i][j] = a.arr[i][j] - b.arr[i][j]
        else:
            for i in range(a.rows):
                for j in range(a.cols):
                    m.arr[i][j] = a.arr[i][j] - b
        return m

    @staticmethod
    def s_mult(a, b):
        m = Matrix(a.rows, a.cols)
        if (isinstance(b, Matrix)):
            for i in range(min(a.rows, b.rows)):
                for j in range(min(a.cols, b.cols)):
                    m.arr[i][j] = a.arr[i][j] * b.arr[i][j]
        else:
            for i in range(a.rows):
                for j in range(a.cols):
                    m.arr[i][j] = a.arr[i][j] * b
        return m

    @staticmethod
    def mat_mul(a, b):
        if (a.cols != b.rows):
            return None
        m = Matrix(a.rows, b.cols)
        for i in range(a.rows):
            for j in range(b.cols):
                for k in range(a.cols):
                    m.arr[i][j] += a.arr[i][k]*b.arr[k][j]
        return m

    @property
    def tr(self):
        m = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                m.arr[j][i] = self.arr[i][j]
        return m
    
    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                self.arr[i][j] = func(self.arr[i][j])
    
    @staticmethod
    def s_map(a, func):
        m = Matrix(a.rows, a.cols)
        for i in range(a.rows):
            for j in range(a.cols):
                m.arr[i][j] = func(a.arr[i][j])
        return m
    
    @staticmethod
    def arg_max(arr):
        max = arr[0]
        index = 0
        for i in range(1, len(arr)):
            if arr[i] > max:
                max = arr[i]
                index = i
        return index; 
    
    @staticmethod
    def one_hot(index, total):
        arr = []
        for i in range(total):
            if i == index:
                arr.append(1)
            else:
                arr.append(0)
        return arr
    
    @staticmethod
    def shuffle(x, y):
        x_copy = x[:]
        y_copy = y[:]
        x_new = []
        y_new = []
        for i in range(len(x)):
            index = random.randint(0, len(x_copy)-1)
            x_new.append(x_copy.pop(index))
            y_new.append(y_copy.pop(index))
        return x_new, y_new
    
    @staticmethod
    def train_test_split(xs, ys, ratio):
        train = round(len(xs)*ratio)
        test = len(xs) - train
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for i in range(train):
            x_train.append(xs[i])
            y_train.append(ys[i])
        for i in range(train, test + train):
            x_test.append(xs[i])
            y_test.append(ys[i])
        return x_train, y_train, x_test, y_test





def f(arr):
    a = arr.copy()
    print(a.pop(2))

if __name__ == "__main__":
    x = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    y = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    x_train, y_train, x_test, y_test = Matrix.train_test_split(x, y, 0.8)
    print(x)
    print(y)
    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)


