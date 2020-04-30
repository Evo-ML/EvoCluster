import matplotlib.pyplot as plt
import pandas as pd 
import objectives

def run(results_directory, optimizer, objectivefunc, dataset_List, Iterations):
    plt.ioff()
    fileResultsData = pd.read_csv(results_directory + '/experiment.csv')

    for d in range(len(dataset_List)):
        dataset_filename = dataset_List[d] + '.csv' 
        for j in range (0, len(objectivefunc)):
            objective_name = objectivefunc[j]

            startIteration = 0                
            if 'SSA' in optimizer:
                startIteration = 2             
            allGenerations = [x+1 for x in range(startIteration,Iterations)]   
            for i in range(len(optimizer)):
                optimizer_name = optimizer[i]

                row = fileResultsData[(fileResultsData["Dataset"] == dataset_List[d]) & (fileResultsData["Optimizer"] == optimizer_name) & (fileResultsData["objfname"] == objective_name)]
                row = row.iloc[:, 19+startIteration:]

                plt.plot(allGenerations, row.values.tolist()[0], label="C" + optimizer_name)
            plt.xlabel('Iterations')
            plt.ylabel('Fitness')
            plt.legend(loc='right')
            plt.grid()
            plt.savefig(results_directory + "/convergence-" + dataset_List[d] + "-" + objective_name + ".pdf")
            plt.clf()
            #plt.show()