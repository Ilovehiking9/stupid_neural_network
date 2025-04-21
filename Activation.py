def relu(num):
    if num <= 0:
        return 0
    else:
        return num

def leakyRelu(num):
    if num < 0:
        return 0.01 * num
    else:
        return num
