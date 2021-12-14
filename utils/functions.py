
def distance_betwen_points(p1, p2, m1, m2, delta_perdas, delta_massas):
    massas = ((m1 - m2) / delta_massas) ** 2
    perdas = ((p1 - p2) / delta_perdas) ** 2

    return (massas + perdas) ** .5


def sum_of_integers(n1, n2):
    return int((n2 - n1 + 1) * (n1 + n2) / 2)
