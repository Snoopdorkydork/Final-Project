'''
***** Final Project 2021 - Software Carpentry *****
Contributors - Mahin Gadkari , Rashi Sultania
Aim:- The aim of the code is to predict the likelyhood of forest fires
given the weather conditions.
'''


class Fire_Predictor():
    def __init__(self):
        self.temp = input('Please enter maximum Temperature in Celcius : ')
        self.RH = input(
            'Please enter Reletive Humidity (RH) percentage value : ')
        self.wind = input('Please enter Wind speed in km/h : ')
        self.rain = input('Please enter rain in mm :')
        self.FFMC = input(
            'Please enter Fine Fuel Moisture Code (FFMC) index : ')
        self.DMC = input(
            'Please enter Duff Moisture Code (DMC) index :')
        self.DC = input(
            'Please enter Drought Code (DC) index :')
        self.ISI = input(
            'Please enter Initial Spread Index (ISI) :')
        self.BUI = input(
            'Please enter Buildup Index (BUI) :')
        self.FWI = input(
            'Please enter Fire Weather Index (FWI) :')



if __name__ == "__main__":
    print("***Welcome to the Forest Fire Predictor***")
    print("Enter the required metereological data as requested")
