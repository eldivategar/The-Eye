import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

pathcsv = 'static/dataRecord/csv/'
pathPieChart = 'static/dataRecord/pieChart/'

def makePieChartFromCsv(filenamePie):
    
    df = pd.read_csv(f'{pathcsv}{filenamePie}.csv')
    
    sum = df['Ekspresi'].value_counts()
    plt.pie(sum, labels=sum.index, autopct='%1.1f%%')
    plt.axis('equal')

    # df['Ekspresi'].value_counts().plot.pie(autopct='%1.1f%%',)
    # plt.axis('equal') 
   
    
    savePie = plt.savefig(pathPieChart+f'{filenamePie}.png', format='png')
    return savePie