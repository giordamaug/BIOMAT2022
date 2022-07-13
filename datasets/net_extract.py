import pandas as pd
import networkx as nx

df_net = pd.read_csv("integratedNet_directed_edges.txt", sep='\t')
df_net.rename(columns = {"from": "source", "to": "target"}, inplace=True) 	# rename from/to as "source/target
df_net.to_csv("integratedNet_edges.csv", index=False)				# save as csv
df_net = pd.read_csv("integratedNet_edges.csv")					# reload csv

# estract and save in csv PPI, MET and MET+PPI networks
df_net["IsPhysical"] = df_net.apply(lambda row: True if row.link == "physical" or row.link == "both" else False, axis=1)
df_net["IsMetabolic"] = df_net.apply(lambda row: True if row.link == "metabolic" or row.link == "both" else False, axis=1)
df_net.loc[df_net["IsMetabolic"]].drop(columns=['IsPhysical', 'IsMetabolic', 'link']).to_csv("met_edges.csv", index=False)
df_net.loc[df_net["IsPhysical"]].drop(columns=['IsPhysical', 'IsMetabolic','link']).to_csv("ppi_edges.csv", index=False)
df_net.drop(columns=['IsPhysical', 'IsMetabolic', 'link']).to_csv("met+ppi_edges.csv")
print(df_net)
