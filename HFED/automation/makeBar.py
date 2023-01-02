import csv
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.style.use("fivethirtyeight")

pathcsv = 'static/dataRecord/csv/'
pathBarChart = 'static/dataRecord/barChart/'
        

def makeBarChart(filenameBar):
    with open(pathcsv+f'{filenameBar}.csv') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        expression_counter = Counter()
        
        for row in csv_reader:
            expression_counter.update(row['Ekspresi'].split(';'))
            
    print(expression_counter)

    expression = []
    total = []

    for item in expression_counter.most_common(15):
        expression.append(item[0])
        total.append(item[1])

    print(expression)
    print(total)  

    plt.figure(figsize=(9, 7))

    plt.bar(expression, total)

    plt.title("Most Expression")

    plt.xlabel("Expression")

    plt.ylabel("Number of Expression")
    saveBar = plt.savefig(f'{pathBarChart}{filenameBar}.png', format='png')

    return saveBar
