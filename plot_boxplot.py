import numpy as np
import pandas as pd 
import plotly.graph_objs as go
import plotly.io as pio

directory = ""
dataset_List = ["diagnosis_II.csv","vary-density.csv","iris.csv","iris2D.csv",
                "wine.csv","seeds.csv","flame.csv","user-level.csv",
                "heart.csv","pathbased.csv","haberman.csv","spiral.csv","ecoli.csv",
                "jain.csv", "mouse.csv", "smiley.csv", "wdbc.csv", "balance.csv",
                "Blood.csv", "aggregation.csv"]
dataset_List = ["aggregation.csv"]


Iterations= 100
fileResultsDetails = "experiment final_details.csv"


optimizer=["PSO","GA","GWO","FFA","CS"]
objectivefunc=["SSE"] 
measure=["SSE", "Purity", "Entropy", "ARI"] 

fileResultsDetailsData = pd.read_csv(directory + fileResultsDetails) 

for d in range(len(dataset_List)):
    for j in range (0, len(objectivefunc)):
        for z in range (0, len(measure)):
            
            #Box Plot
            data = []
            for i in range(len(optimizer)):            
                detailedData = fileResultsDetailsData[(fileResultsDetailsData["Dataset"] == dataset_List[d]) & (fileResultsDetailsData["Optimizer"] == optimizer[i]) & (fileResultsDetailsData["objfname"] == objectivefunc[j])]
                detailedData = detailedData[measure[z]]
                detailedData = np.array(detailedData).T.tolist()
                data.append(go.Box(      
                    y=detailedData,
                    name = "C" + optimizer[i],
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
            pio.write_image(fig, "plot/boxplot/boxplot-" + dataset_List[d].replace('.csv', '') + "-" + objectivefunc[j] + "-" + measure[z] + ".pdf")
            