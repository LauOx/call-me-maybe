import aux


if __name__ == "__main__":
    number = '4.00000000'
    try:
        print(f"{aux.clean_number(number)} jeje")
    except ValueError:
        print("no se puede por lo que sea")