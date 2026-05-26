def get_patient_info():
    name = input("enter your name: ")
    while True:
        try:
            age= int(input("enter your age: "))
            if age > 0:
                break
        except ValueError:
            print("Not Valid - Age must be a number")

    while True:
        try:
            weight = float(input("enter your weight: "))
            if weight > 0:
                break
        except ValueError:
            print("Not Valid - weight must be a number")
    return name, age, weight
def get_drug_info():
    drug_name = input("enter drug name: ")
    while True:
        try:
            dose = float(input("enter the dose: "))
            if dose > 0:
                break
        except ValueError:
            print("Not Valid - Enter appropiate dose")

    while True:
        try:
            F = float(input("enter the F: "))
            if F > 0:
                break
        except ValueError:
            print("Not Valid - it must be a number")

    while True:
        try:
            vc = float(input("enter vc: "))
            if vc > 0:
                break
        except ValueError:
            print("Not Valid - please try number")

    while True:
        try:
            half_life = float(input("enter half_life: "))
            if half_life > 0:
                break
        except ValueError:
            print("Not Valid - please try number")

    while True:
        try:
            cl = float(input("enter cl: "))
            if cl > 0:
                break
        except ValueError:
            print("Not Valid - please try number")

    while True:
        try:
            t = float(input("enter time: "))
            if t > 0:
                break
        except ValueError:
            print("Not Valid - please try number")

    while True:
        try:
            t_min = float(input("enter therapeutic range min (mg/l): "))
            t_max = float(input("enter therapeutic range max (mg/l): "))
            if t_max > t_min:
                break
        except ValueError:
            print("Not valid- please enter number")
    return drug_name, dose, F, vc, half_life, cl, t_min, t_max
#k = elimination rate constant (how fast the drug leave body)
#c0 = initial concentration after dose
#ct = concentration at time t