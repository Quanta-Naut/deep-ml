import math

def softmax(scores: list[float]) -> list[float]:
    result = []
    base = 0.0
    c = max(scores)

    for i in scores:
        base += math.exp(i - c)

    for i in scores:
        result.append(math.exp(i - c)/base)
    
    return result