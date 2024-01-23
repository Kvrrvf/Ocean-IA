### Principal Component Analysis of OM-RGC_v2 metagenomic gene profiles ###

print("")
print("             PCA.py script started")

# Libraries

#!sudo-g5k pip install -U scikit-learn plotly xlsx2csv kaleido
print("Importing libraries...")
import time
from datetime import timedelta
import polars as pl
import polars.selectors as cs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

t0=time.time()
def T():
    return str(timedelta(seconds=round(time.time()-t0)))

# Read data

print("Importing data...")
m = pl.read_excel(source="~/Ocean-IA/group_storage/TARA/OM_RGC_V2/Salazar_et_al_2019_Suppl_Info.xlsx",
                  sheet_name="Table_W4",
                  read_csv_options={"null_values": ['NA'], "n_rows": 180})
m = m.rename({"PANGAEA sample id":"PANGAEA.sample.id"})
print(f"Environmental dataframe dimensions: {m.shape}")
 
metaG = pl.read_csv('~/Ocean-IA/group_storage/TARA/OM_RGC_V2/OM-RGC_v2_gene_profile_metaG.tsv.gz',
                    separator='\t',
                    dtypes=([pl.Utf8]+[pl.Float64]*180))
print(f"MetaG dataframe dimensions: {metaG.shape}")
print(f"Imported data. Time = {T()}")

# Processing data

print("Processing data...")
s_t = metaG.transpose(include_header=True, column_names=metaG["OMRGC_ID"])
print(f"Check tranpose. Time = {T()}")
s_t = s_t.slice(1)
print(f"Check slice. Time = {T()}")
s_t = s_t.cast({cs.starts_with("OM"):pl.Float64})
print(f"Check dtype changes. Time = {T()}")
s_t = s_t.rename({"column":"PANGAEA.sample.id"})
print(f"Check rename. Time = {T()}")
print("New dataframe dimensions:", s_t.shape)

# MIX layer filtering

print("Filtering data...")
MIX = m.filter(pl.col("Layer") == "MIX")
print("MIX df dimensions:", MIX.shape)
L = MIX["PANGAEA.sample.id"].to_list()
s_t = s_t.filter(~pl.col("PANGAEA.sample.id").is_in(L))
print("Filtered metaG df dimensions:", s_t.shape)
print(f"Filtering time = {T()}")

# Normalization

print("Normalizing data...")
n = s_t.drop("PANGAEA.sample.id")
N = StandardScaler().fit_transform(n)
print(f"Normalization time = {T()}")

print("PCA...")
batch_size = N.shape[0]
ipca = IncrementalPCA(n_components=None, batch_size=batch_size)

for i in range(0, N.shape[0], batch_size):
    ipca.partial_fit(N[i:i+batch_size])
print(f"PCA time = {T()}")

figure(figsize=(8,6), dpi=100)
plt.plot(np.cumsum(ipca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.axvline(x=3, c="red", ls='--')
plt.title('IPCA meta-G subset', size=18)
explained_var = np.cumsum(ipca.explained_variance_ratio_)
plt.text(10, explained_var[3], f'Explained Variance at x=3: {explained_var[3]*100:.2f}%', color='red')
plt.savefig('PCA_explained_var.png')
print(f"PCA explained variance figure saved. Time = {T()}")

ipca2 = IncrementalPCA(n_components=2, batch_size=batch_size)
components = ipca2.fit_transform(N)
print(f"PCA 2D time = {T()}")

labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(ipca2.explained_variance_ratio_ * 100)
    }
fig = px.scatter(components, x=0, y=1, width=800, height=600,
                 color=s_tj['Layer'],
                 labels=labels)
fig.update_traces(marker_size=8)
fig.update_layout(legend_itemsizing="constant",  legend_font_size=20, legend_title="Layer", font=dict( size=16))
fig.write_html('PCA_2D.html')
fig.write_image('PCA_2D.png')
print(f"PCA 2D figure saved. Time = {T()}")

ipca3 = IncrementalPCA(n_components=3, batch_size=batch_size)
components = ipca3.fit_transform(N)
print(f"PCA 3D time = {T()}")

labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(ipca3.explained_variance_ratio_ * 100)
    }
fig = px.scatter_3d(components, x=0, y=1, z=2, color=s_tj['Layer'],
                    labels=labels,width=800, height=600,opacity=.4,template="plotly_white")
fig.update_traces(marker_size=8)
fig.update_layout(legend_itemsizing="constant",  legend_font_size=20, legend_title="Layer", font=dict( size=16))
fig.write_html('PCA_3D.html')
fig.write_image('PCA_3D.png')
print(f"PCA 3D figure saved. Time = {T()}")
print(f"            PCA.py script finished. Time = {T()}")
print("")