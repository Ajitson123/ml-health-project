with open("names.csv") as f:
    for el in f:
        v = el.rstrip().split(",")
        print(f"{v[0]} is in {v[1]}")