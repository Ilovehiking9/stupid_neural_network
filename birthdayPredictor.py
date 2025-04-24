from Network import Network
import datetime


now = datetime.date.today()


def getYear():
    return(int(input("What year were you born? ")))

def getMonth():
    monthList = ["january", "february", "march", "april", "may", "june", 'july', "august", "september", "october", "november", "december"]

    userInput = input("What month were you born? ")
    try:
        return(int(userInput))
    except ValueError:
        for i, month in enumerate(monthList):
            if month in userInput:
                return(i + 1)

def getDay():
    return(int(input("What day were you born? ")))


print("im gonna guess ur birthday")
year = getYear()
month = getMonth()
day = getDay()

normalYear = (year-1900) / (130)
normalMonth = month / 12
normalDay = day/31

neuralNetwork = Network()

neuralNetwork.addInputLayer(6)
neuralNetwork.addLayer(8)
neuralNetwork.addLayer(4)
neuralNetwork.addLayer(2)

neuralNetwork.init()

neuralNetwork.loadWeightsAndBiases("weights_and_biases.json")

neuralNetwork.forward([normalMonth, normalDay, normalYear, now.month/12, now.day/31, (now.year-1900)/130])

old = []
for node in neuralNetwork.layers[-1].nodes:
    old.append(node.output)

print(f"you are {int(old[0] * 130)} years old")
print("unc.")

