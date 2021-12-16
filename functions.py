'''
***** Final Project 2021 - Software Carpentry *****
Contributors - Mahin Gadkari , Rashi Sultania
Aim:- The aim of the code is to predict the likelyhood of forest fires
given the weather conditions.
'''


def get_inputs():
    while True:
        temp = input('Please enter Temperature in Celcius : ')
        try:
            temp = float(temp)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        RH = input(
            'Please enter Reletive Humidity (RH) percentage value : ')
        try:
            RH = float(RH)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        wind = input('Please enter Wind speed in km/h : ')
        try:
            wind = float(wind)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        rain = input('Please enter rain in mm : ')
        try:
            rain = float(rain)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        FFMC = input(
            'Please enter Fine Fuel Moisture Code (FFMC) index : ')
        try:
            FFMC = float(FFMC)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        DMC = input(
            'Please enter Duff Moisture Code (DMC) index : ')
        try:
            DMC = float(DMC)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        ISI = input(
            'Please enter Initial Spread Index (ISI) : ')
        try:
            ISI = float(ISI)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        BUI = input(
            'Please enter Buildup Index (BUI) : ')
        try:
            BUI = float(BUI)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        FWI = input(
            'Please enter Fire Weather Index (FWI) : ')
        try:
            FWI = float(FWI)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        DC = input(
            'Please enter Drought Code (DC) index : ')
        try:
            DC = float(DC)
            break
        except ValueError:
            print("Enter a valid value!")

    input_val = [temp, RH, wind, rain, FFMC, DMC, DC, ISI, BUI, FWI]
    return input_val


if __name__ == "__main__":
    print("***Welcome to the Forest Fire Predictor***")
    print("Enter the required metereological data as requested")
    inputs = get_inputs()
    print(inputs)
