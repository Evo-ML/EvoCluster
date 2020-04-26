import matplotlib.pyplot as plt
import pandas as pd 
import objectives

def run(results_directory, optimizer, objectivefunc, dataset_List, Iterations):

    fileResultsData = pd.read_csv(results_directory + '/experiment.csv')

    for d in range(len(dataset_List)):
        for j in range (0, len(objectivefunc)):
            if(objectivefunc[j]==True):
                #Convergence Curve
                startIteration = 0                
                if optimizer[0]==True:#CSSA
                    startIteration = 2             
                allGenerations = [x+1 for x in range(startIteration,Iterations)]   
                for i in range(len(optimizer)):
                    if(optimizer[i]==True):
                        objective_name = objectives.getFunctionDetails(j)[0]
                        optimizer_dict = {0:'SSA', 1:'PSO', 2:'GA', 3:'BAT', 4:'FFA', 5:'GWO', 6: 'WOA', 7:'MVO', 8:'MFO',9:'CS'}
                        optimizer_name = optimizer_dict.get(i)
                        #print(optimizer_name)
                        row = fileResultsData[(fileResultsData["Dataset"] == dataset_List[d]) & (fileResultsData["Optimizer"] == optimizer_name) & (fileResultsData["objfname"] == objective_name)]
                        row = row.iloc[:, 19+startIteration:]
                        #print(row.values.tolist())
                        plt.plot(allGenerations, row.values.tolist()[0], label="C" + optimizer_name)
                plt.xlabel('Iterations')
                plt.ylabel('Fitness')
                plt.legend(loc='right')
                plt.grid()
                plt.savefig(results_directory + "/convergence-" + dataset_List[d].replace('.csv', '') + "-" + objective_name + ".pdf")
                plt.clf()
                #plt.show()

