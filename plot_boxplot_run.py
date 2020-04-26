import numpy as np
import pandas as pd 
import plotly.graph_objs as go
import plotly.io as pio
import objectives

def run(results_directory, optimizer, objectivefunc, dataset_List, Iterations):

    measure=['SSE','Purity','Entropy', 'HS', 'CS', 'VM', 'AMI', 'ARI', 'Fmeasure', 'TWCV', 'SC', 'Accuracy', 'DI', 'DB', 'STDev']

    fileResultsDetailsData = pd.read_csv(results_directory + '/experiment_details.csv')
    for d in range(len(dataset_List)):
        for j in range (0, len(objectivefunc)):

            if(objectivefunc[j]==True):
                for z in range (0, len(measure)):
                    
                    #Box Plot
                    data = []      
                        
                    for i in range(len(optimizer)):                
                        if(optimizer[i]==True):
                            objective_name = objectives.getFunctionDetails(j)[0]
                            optimizer_dict = {0:'SSA', 1:'PSO', 2:'GA', 3:'BAT', 4:'FFA', 5:'GWO', 6: 'WOA', 7:'MVO', 8:'MFO',9:'CS'}
                            optimizer_name = optimizer_dict.get(i)          
                            
                            detailedData = fileResultsDetailsData[(fileResultsDetailsData["Dataset"] == dataset_List[d]) & (fileResultsDetailsData["Optimizer"] == optimizer_name) & (fileResultsDetailsData["objfname"] == objective_name)]
                            detailedData = detailedData[measure[z]]
                            detailedData = np.array(detailedData).T.tolist()
                            data.append(go.Box(      
                                y=detailedData,
                                name = "C" + optimizer_name,
                            ))
                            
                    layout = go.Layout(
                        legend=dict(font=dict(size=18)),
                        font=dict(size=22),
                        #yaxis = dict(range=[0,1],showline=True),
                        yaxis = dict(showline=True),
                        xaxis = dict(showline=True),
                        showlegend=True
                    )
                    
                    fig = go.Figure(data=data, layout=layout)
                    pio.write_image(fig, results_directory + "/boxplot-" + dataset_List[d].replace('.csv', '') + "-" + objective_name + "-" + measure[z] + ".pdf")
                    


