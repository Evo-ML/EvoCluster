import matplotlib.pyplot as plt
import pandas as pd 

directory = ""
dataset_List = ["diagnosis_II.csv","vary-density.csv","iris.csv","iris2D.csv",
                "wine.csv","seeds.csv","flame.csv","user-level.csv",
                "heart.csv","pathbased.csv","haberman.csv","spiral.csv","ecoli.csv",
                "jain.csv", "mouse.csv", "smiley.csv", "wdbc.csv", "balance.csv",
                "Blood.csv", "aggregation.csv"]
dataset_List = ["aggregation.csv", "seeds.csv", "flame.csv","iris.csv"]


Iterations= 100
fileResults = "experiment final.csv"


optimizer=["PSO","GA","GWO","FFA","CS"]
objectivefunc=["SSE"] 

fileResultsData = pd.read_csv(directory + fileResults) 

for d in range(len(dataset_List)):
    for j in range (0, len(objectivefunc)):
        #Convergence Curve
        startIteration = 0                
        if optimizer[0]==True:#CSSA
            startIteration = 2             
        allGenerations = [x+1 for x in range(startIteration,Iterations)]   
        for i in range(len(optimizer)):
            row = fileResultsData[(fileResultsData["Dataset"] == dataset_List[d]) & (fileResultsData["Optimizer"] == optimizer[i]) & (fileResultsData["objfname"] == objectivefunc[j])]
            row = row.iloc[:, 19+startIteration:]
            plt.plot(allGenerations, row.values.tolist()[0], label="C" + optimizer[i])
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.legend(loc='right')
        plt.grid()
        plt.savefig("plot/convergence/convergence-" + dataset_List[d].replace('.csv', '') + "-" + objectivefunc[j] + ".pdf")
        plt.clf()
        #plt.show()
        