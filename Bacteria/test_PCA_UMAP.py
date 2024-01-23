print("")
print("### Starting dimensional reduction script ###")
print("")
print("Loading libraries...")

import time
from datetime import timedelta
import polars as pl
import polars.selectors as cs

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.pyplot import figure
from sklearn.preprocessing import StandardScaler

import umap.umap_ as umap
from umap import UMAP
import hdbscan

print("Libraries loaded successfully")
print("")
print("Reading dataframes...")

np.random.seed(1)

t0 = time.time()

m = pl.read_excel(source="/home/npilquinao/TARA/bacteria/data/Salazar_et_al_2019_Suppl_Info.xlsx",
                  sheet_name="Table_W4",
                  read_csv_options={"null_values": ['NA'], "n_rows": 180})
m = m.rename({"PANGAEA sample id":"PANGAEA.sample.id"})
m = m[["PANGAEA.sample.id","Layer"]]

print(f"   Layer info from enviromental dataset {m.shape}")

s = pl.read_parquet("/home/npilquinao/TARA/bacteria/data/metaG_v1_lz4.parquet",
                    n_rows=1000000)

print(f"   MetaG subset {s.shape}")

t1 = time.time()

print(f"Dataframes has been read correctly. Time: {str(timedelta(seconds=round(t1-t0)))}")
print("")
print("Tranpose starting...")

s_t = s.transpose(include_header=True, column_names=s["OMRGC_ID"])
s_t = s_t.slice(1)
s_t = s_t.cast({cs.starts_with("OM"):pl.Float64})
s_t = s_t.rename({"column":"PANGAEA.sample.id"})

t2 = time.time()

print(f"Tranpose ready. New MetaG dataframe dimensions: {s_t.shape}. Time: {str(timedelta(seconds=round(t2-t0)))}")
print("")
print("Starting join...")

s_tj = s_t.join(m, on="PANGAEA.sample.id", how="left")

t3 = time.time()

print(f"Join ready. New dataframe dimensions: {s_tj.shape}. Time: {str(timedelta(seconds=round(t3-t0)))}")
print("")
print("Starting data normalization...")

n = s_t.drop("PANGAEA.sample.id")
N = StandardScaler().fit_transform(n)

t4 = time.time()

print(f"Data has been normalized. Time: {str(timedelta(seconds=round(t4-t0)))}")
print("")
print("Starting PCA...")

pca = PCA().fit(N)
figure(figsize=(8,6), dpi=200)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.axvline(x=3, c="red", ls='--')
plt.title('PCA meta-G subset', size=18)
explained_var = np.cumsum(pca.explained_variance_ratio_)
plt.text(10, explained_var[3], f'Explained Variance at x=3: {explained_var[3]*100:.2f}% ; x=2: {explained_var[2]*100:.2f}%',
         color='red')

plt.savefig("/home/npilquinao/TARA/bacteria/data/test_PCA_cumulativeVariance.png", facecolor="white")

t5 = time.time()

print(f"   PCA cumulative variance plot ready and exported. Time: {str(timedelta(seconds=round(t5-t0)))}")

pca = PCA(n_components=2)
components = pca.fit_transform(N)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
fig = px.scatter(components, x=0, y=1, width=1000, height=800,
                 color=s_tj['Layer'],
                 labels=labels)
fig.update_traces(marker_size=8)
fig.update_layout(legend_itemsizing="constant",  legend_font_size=20, legend_title="Layer", font=dict( size=16))

fig.write_html("/home/npilquinao/TARA/bacteria/data/2_components_PCA.html")

t6 = time.time()

print(f"   2-PC plot ready and exported. Time: {str(timedelta(seconds=round(t6-t0)))}")

pca = PCA(n_components=3)
components = pca.fit_transform(N)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
fig = px.scatter_3d(components, x=0, y=1, z=2, color=s_tj['Layer'],
                    labels=labels,width=1000, height=800,opacity=.4,template="plotly_white")
fig.update_traces(marker_size=8)
fig.update_layout(legend_itemsizing="constant",  legend_font_size=20, legend_title="Layer", font=dict( size=16))

fig.write_html("/home/npilquinao/TARA/bacteria/data/3_components_PCA.html")

t7 = time.time()

print(f"   3-PC plot ready and exported. Time: {str(timedelta(seconds=round(t7-t0)))}")
print("")
print("Starting UMAP...")





print("Finish. No errors.")
print("")