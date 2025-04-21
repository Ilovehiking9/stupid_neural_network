import csv

def loadDataset(filepath):
    dataset = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            # First 6 columns are inputs, last 2 columns are expected outputs
            inputs = list(map(float, row[:6]))
            expected_outputs = list(map(float, row[6:]))
            dataset.append((inputs, expected_outputs))
    return dataset