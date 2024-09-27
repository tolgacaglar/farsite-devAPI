from matplotlib import pyplot as plt
from shapely import Polygon, MultiPolygon, GeometryCollection, make_valid

import geopandas as gpd
import pandas as pd
import numpy as np
import os

###### MISC ###########
def calculate_max_area_geom(multigeom):
    if isinstance(multigeom, GeometryCollection) | isinstance(multigeom, MultiPolygon):
        max_area = 0
        max_area_idx = 0
        for ix, g in enumerate(multigeom.geoms):
            if g.area > max_area:
                max_area = g.area
                max_area_idx = ix
        return calculate_max_area_geom(multigeom.geoms[max_area_idx])
    
    return multigeom

def validate_geom(poly):
    poly = make_valid(poly)
    if isinstance(poly, GeometryCollection) | isinstance(poly, MultiPolygon):
        poly = calculate_max_area_geom(poly)
    
    if not isinstance(poly, Polygon):
        print('buffered polygon is not a polygon..')
        print(f'type = {type(poly)}')
        
#     assert(isinstance(poly, Polygon)), 'buffered polygon is not a polygon'
    
    return poly


###### Data
def get_observation(description : str, tix : int):
    ''' Obtain selected observation geometry and datetime
    description: ['Maria2019', 'River2021', 'Bridge2021', 'CA-FKU-BOLT', 'CA-FKU-FLASH', 'CA-LAC-POST', 'CA-SCU-CORRAL']
    '''

    if description in ['Maria2019', 'River2021', 'Bridge2021']:
    
        df = pd.read_pickle(os.path.join(os.getenv('HOME'), 'farsite-devAPI', 'data', 'dftable_06032023.pkl'))
        df['filepath'] = df['filepath'].str[0:14] + 'farsite-devAPI' + df['filepath'].str[21:]

        dfrow = df[df['description'] == description].sort_values('datetime').iloc[tix]
        dfgeom = gpd.read_file(dfrow['filepath'])['geometry'][0]
        dfdt = dfrow['datetime']
    elif description in ['CA-FKU-BOLT', 'CA-FKU-FLASH', 'CA-LAC-POST', 'CA-SCU-CORRAL']:
        gdf = gpd.read_file('/home/tcaglar/farsite-devAPI/data/intterra_firis.geojson')
        def fill_json(kw):
            if kw['json']['source'] == None:
                return ''
            return kw['json']['source']

        gdf['source'] = gdf.apply(fill_json, axis=1)
        
        gdfFiltered = gdf[gdf['source'].str.contains(description)].sort_values('created_time').to_crs(epsg=5070)
        gdfFiltered['created_time'] = pd.to_datetime(gdfFiltered['created_time'])
        
        dfgeom = gdfFiltered['geometry'].iloc[tix]
        dfdt = gdfFiltered['created_time'].iloc[tix]

    else:
        raise ValueError(f'description {description} not present in db')
        
    return dfgeom, dfdt


##### PLOTTING
###############
def plot_geometry(geom, ax = None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(4,4))
    
    if isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            x,y = g.exterior.coords.xy
            ax.plot(x,y, **kwargs)
    else:
        x,y = geom.exterior.coords.xy
        ax.plot(x,y, **kwargs)
        
    ax.set_aspect('equal')
        
def plot_matrix(X, ax=None, show_stdev = False, **kwargs):
    vcounts = X.shape[0]//2
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(4,4))
    
    color = (1,0,0,0.9)
    if 'color' in kwargs:
        color = kwargs['color']
        
    X_std = np.std(X, axis=1)
    X_mean = np.mean(X, axis=1)
    ax.plot(X_mean[::2], X_mean[1::2], **kwargs)

    # Calculate standard deviation of the generated coordinates
    x0, y0 = X_mean[::2], X_mean[1::2]
    radstd = np.zeros_like(x0)
    
    if show_stdev:
        for vix in range(vcounts):
            print(f'Calculating {vix}/{vcounts}..    ', end='\r', flush=True)
            x,y = X[2*vix,:], X[2*vix+1,:]
            radius = np.sqrt((x-x0[vix])**2 +(y-y0[vix])**2)
            radstd[vix] = np.std(radius)
        print()
        for vix in range(vcounts):
            print(f'Drawing {vix}/{vcounts}..    ', end='\r', flush=True)
            circle = plt.Circle((x0[vix], y0[vix]), radius=radstd[vix], fill=False, edgecolor=(0,0,0,0.4), lw=0.3)
            ax.add_artist(circle)
            
    ax.set_aspect('equal')

def plot_matrix_ensemble(X, ax=None, plot_alix = None, alpha=0.1, **kwargs):
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(4,4))
        
    for vix in range(X.shape[1]):
        ax.plot(X[::2, vix], X[1::2, vix], **kwargs)
        
    if plot_alix is not None:
        ax.scatter(X[2*plot_alix,:], X[2*plot_alix+1, :], alpha = alpha, color=kwargs['color'], edgecolors=(0,0,0,0))
    ax.set_aspect('equal')
