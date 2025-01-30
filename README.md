# Modified_Pivot_Correlation_Clustering

The following code generates the Sthocastic Block Models as an example. 
Given a graph G, by creating class CorrelationClustering a random permutatuion over vertices of the graph will be picked.
This permutation will be used for calculating clusters of pivot. 
As a postprocessing to pivot algorithm, Modified_Pivot will refine this clusterings to further improve the objective value.
Function get_disagreement will output the disagreements by pivot and modified_pivot seperately.

