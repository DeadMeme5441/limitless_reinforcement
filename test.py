import math


def poisson(i, l):
    return ((math.e ** (-l)) * (l**i)) / (math.factorial(i))


if __name__ == "__main__":
    for i in range(10):
        print(poisson(i, 1.0))
