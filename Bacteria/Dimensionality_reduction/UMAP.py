### Uniform Manifold Approximation and Projection for Dimension Reduction of OM-RGC_v2 metagenomic gene profiles ###

print("")
print("             UMAP.py script started")

# Libraries

#!sudo-g5k pip install -U scikit-learn umap-learn plotly xlsx2csv kaleido
print("Importing libraries...")
import time
from datetime import timedelta
import polars as pl
import polars.selectors as cs
import numpy as np
import plotly.express as px
import plotly.io as py
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from umap import UMAP
import hdbscan

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
m = m.filter(~pl.col("PANGAEA.sample.id").is_in(L))
print("Filtered environmental df dimensions:", m.shape)
print(f"Filtering time = {T()}")

# Normalization

print("Normalizing data...")
n = s_t.drop("PANGAEA.sample.id")
N = StandardScaler().fit_transform(n)
print(f"Normalization time = {T()}")

# UMAP

print("UMAP processing...")
umap_model_2D = umap.UMAP(n_neighbors=40, n_components=2, min_dist=0.1, spread=1.0, n_jobs=-1)
umap_2D = umap_model_2D.fit_transform(N)

figure_2D = px.scatter(
    umap_2D, x=0, y=1, color=m['Layer'],
    labels={"0": "Dimension 1", "1": "Dimension 2"},
    template="plotly_white", width=1000, height=800, opacity=.6)

figure_2D.update_traces(marker_size=8)
figure_2D.update_layout(legend_itemsizing="constant", legend_font_size=15, font=dict(size=12))
py.write_html(figure_2D, 'UMAP_2D_layer.html')
py.write_image(figure_2D, 'UMAP_2D_layer.png', scale=2)

figure_2D = px.scatter(
    umap_2D, x=0, y=1, color=m['polar'],
    labels={"0": "Dimension 1", "1": "Dimension 2"},
    template="plotly_white", width=1000, height=800, opacity=.6)

figure_2D.update_traces(marker_size=8)
figure_2D.update_layout(legend_itemsizing="constant", legend_font_size=15, font=dict(size=12))
py.write_html(figure_2D, 'UMAP_2D_polar.html')
py.write_image(figure_2D, 'UMAP_2D_polar.png', scale=2)

figure_2D = px.scatter(
    umap_2D, x=0, y=1, color=m['Ocean.region'],
    labels={"0": "Dimension 1", "1": "Dimension 2"},
    template="plotly_white", width=1000, height=800, opacity=.6)

figure_2D.update_traces(marker_size=8)
figure_2D.update_layout(legend_itemsizing="constant", legend_font_size=15, font=dict(size=12))
py.write_html(figure_2D, 'UMAP_2D_ocean_region.html')
py.write_image(figure_2D, 'UMAP_2D_ocean_region.png', scale=2)

print("UMAP 2D figures saved")
print(f"UMAP 2D process time = {T()}")

umap_model_3D = umap.UMAP(n_neighbors=40,n_components=3, min_dist=0.1, spread=1.0, n_jobs=-1)
umap_3D = umap_model_3D.fit_transform(N)

figure_3D = px.scatter_3d(
    umap_3D, x=0, y=1, z=2, color=m['Layer'],
    labels={"0": "Dimension 1", "1": "Dimension 2", "2": "Dimension 3"},
    template="plotly_white", width=1000, height=800, opacity=.5)

figure_3D.update_traces(marker_size=11)
figure_3D.update_layout(legend_itemsizing="constant", legend_font_size=10, font=dict(size=12))
py.write_html(figure_3D, 'UMAP_3D_layer.html')
py.write_image(figure_3D, 'UMAP_3D_layer.png', scale=2)

figure_3D = px.scatter_3d(
    umap_3D, x=0, y=1, z=2, color=m['polar'],
    labels={"0": "Dimension 1", "1": "Dimension 2", "2": "Dimension 3"},
    template="plotly_white", width=1000, height=800, opacity=.5)

figure_3D.update_traces(marker_size=11)
figure_3D.update_layout(legend_itemsizing="constant", legend_font_size=10, font=dict(size=12))
py.write_html(figure_3D, 'UMAP_3D_polar.html')
py.write_image(figure_3D, 'UMAP_3D_polar.png', scale=2)

figure_3D = px.scatter_3d(
    umap_3D, x=0, y=1, z=2, color=m['Ocean.region'],
    labels={"0": "Dimension 1", "1": "Dimension 2", "2": "Dimension 3"},
    template="plotly_white", width=1000, height=800, opacity=.5)

figure_3D.update_traces(marker_size=11)
figure_3D.update_layout(legend_itemsizing="constant", legend_font_size=10, font=dict(size=12))
py.write_html(figure_3D, 'UMAP_3D_ocean_region.html')
py.write_image(figure_3D, 'UMAP_3D_ocean_region.png', scale=2)

print("UMAP 3D figures saved")
print(f"UMAP 3D process time = {T()}")

print(f"            UMAP.py script finished. Time = {T()}")
print("")