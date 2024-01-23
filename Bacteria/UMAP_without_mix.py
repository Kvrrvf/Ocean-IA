### UMAP OM-RGC v2 without mix samples ###

#sudo-g5k pip install -U scikit-learn umap-learn plotly xlsx2csv

import polars as pl
import polars.selectors as cs
import time
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.pyplot import figure
from sklearn.preprocessing import StandardScaler

import umap.umap_ as umap
from umap import UMAP
import hdbscan

t0=time.time()

metaG = pl.read_csv('~/Ocean_IA//OM_RGC_V2/OM-RGC_v2_gene_profile_metaG.tsv.gz',
                    separator='\t',
                    dtypes=([pl.Utf8]+[pl.Float64]*180),
                    n_rows=1000000)

t1=time.time()
print(f"Read metaG. Time = {str(timedelta(seconds=round(t1-t0)))}")
print(f"    Dataframe dimensions: {metaG.shape}")

m = pl.read_excel(source="/home/npilquinao/TARA/bacteria/data/OM_RGC_V2/Salazar_et_al_2019_Suppl_Info.xlsx",
                  sheet_name="Table_W4",
                  read_csv_options={"null_values": ['NA'], "n_rows": 180})
m = m.rename({"PANGAEA sample id":"PANGAEA.sample.id"})

t2=time.time()
print(f"Read metadata. Time = {str(timedelta(seconds=round(t2-t0)))}")
print(f"    Dataframe dimensions: {m.shape}")

s_t = metaG.transpose(include_header=True, column_names=metaG["OMRGC_ID"])

t3= time.time()
print(f" Check tranpose. Time = {str(timedelta(seconds=round(t3-t0)))}")

s_t = s_t.slice(1)

t4=time.time()
print(f" Chek slice. Time = {str(timedelta(seconds=round(t4-t0)))}")

s_t = s_t.cast({cs.starts_with("OM"):pl.Float64})

t5=time.time()
print(f" Check dtype changes. Time = {str(timedelta(seconds=round(t5-t0)))}")

s_t = s_t.rename({"column":"PANGAEA.sample.id"})

t6=time.time()
print(f" Check rename. Time = {str(timedelta(seconds=round(t6-t0)))}")
print("Success. New dataframe dimensions:", s_t.shape)

t0=time.time()

n = s_t.drop("PANGAEA.sample.id")

t1=time.time()

print(f"Time = {str(timedelta(seconds=round(t1-t0)))}")
print(f"Dataframe dimensions: {n.shape}")

t0=time.time()

N = StandardScaler().fit_transform(n)

t1=time.time()

print(f"Normalization time = {str(timedelta(seconds=round(t1-t0)))}")

t0=time.time()

umap_model = umap.UMAP(n_neighbors=40,n_components=3, min_dist=0.1, spread=1.0,random_state=42)
umap_3D = umap_model.fit_transform(N)

t1=time.time()

print(f"Process time = {str(timedelta(seconds=round(t1-t0)))}")

figure = px.scatter_3d(
    umap_3D, x=0, y=1, z=2, color=m['Ocean.region'],
    labels={"0": "Dimension 1", "1": "Dimension 2", "2": "Dimension 3"},
    template="plotly_white", width=800, height=600, opacity=.5)

figure.update_traces(marker_size=11)
figure.update_layout(legend_itemsizing="constant", legend_font_size=10, font=dict(size=12))
figure.show()

