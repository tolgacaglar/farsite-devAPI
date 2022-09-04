import datetime
import pandas as pd
import geopandas as gpd
from dataclasses import dataclass
import uuid
from multiprocessing import Pool

from shapely.geometry import MultiPolygon, Polygon

import ipywidgets
from ipywidgets import IntRangeSlider, IntSlider, SelectionRangeSlider, SelectionSlider, VBox, HBox, Button, Dropdown
from ipywidgets import Layout, FloatProgress, HBox, VBox, FloatText, Label, IntProgress

from ipyleaflet import Map, basemaps, basemap_to_tiles, ScaleControl, ZoomControl, LayersControl, WKTLayer, WidgetControl
from ipyleaflet.leaflet import LayerException

from functools import partial

import os

@dataclass
class Input:
    startdt: datetime.datetime
    enddt: datetime.datetime
    deltadt: datetime.timedelta
        
    igniteidx: str
    compareidx: str
    lcpidx: str
    barrieridx: str
        
    description: str
        
    windspeed_lst: list
    winddirection_lst: list
        
    temperature: int
    humidity: int

class FilePaths:
    def __init__(self, datadir):
        self.datadir = datadir
        self.dfpath = os.path.join(self.datadir, 'test_table.pkl')
    
    def create_rundir(self):
        dtdir = datetime.datetime.now().strftime('%Y%m%d')
        self.basedir = os.path.join(self.datadir, dtdir)
        
        # If dtdir does not exist, make one
        if not os.path.isdir(self.basedir):
            os.mkdir(self.basedir)
        
        # Find the non-existing folder
        isdirfound = False
        for cnt in range(100000):
            rundir = os.path.join(self.basedir, 'Run_{:05d}'.format(cnt))
            if not os.path.isdir(rundir):
                isdirfound = True
                break
        
        if isdirfound:
            os.mkdir(rundir)
            return rundir
        print('Max iteration reached! {}. No empty dir found'.format(cnt))
        
class Database:
    def __init__(self, fp: FilePaths):
        # Setup params
        self.fp = fp
        
        # TODO
        # Setup the database for reading
        
        
        try:
            dftable = pd.read_pickle(self.fp.dfpath)
        except FileNotFoundError:
            print(f'\n!!Caution!! Path {self.fp.dfpath} not found! Cannot choose ignition!!\n')
            raise
            
        # Collect the tables in dataframe format
        # Table 1 - ignition
        self.gdfignitionAll = gpd.GeoDataFrame(dftable[dftable['filetype'] == 'Ignition'])
        for (idx, ignition) in self.gdfignitionAll.iterrows():
            geom = gpd.read_file(ignition['filepath']).loc[0,'geometry']
            self.gdfignitionAll.loc[idx, 'shape'] = geom.to_wkb()

        gs = gpd.GeoSeries.from_wkb(self.gdfignitionAll['shape'])
        self.gdfignitionAll['geometry'] = gs
        self.gdfignitionAll = self.gdfignitionAll.drop(columns='shape').set_crs(epsg=5070).to_crs(epsg=4326)
        
#         self.gdfignition['description'] = 'Maria2019'
        
        # Table 2 - barrier
        self.dfbarrier = dftable[dftable['filetype'] == 'Barrier'][['filetype', 'filepath']]
        # Table 3 - landscape
        self.dflandscapeAll = dftable[dftable['filetype'] == 'Landscape'][['filetype', 'filepath', 'description']]
        # Table 4 - simulation
        self.gdfsimulation = gpd.GeoDataFrame()
        
        self.filter_selection('Maria2019')
        
    def filter_selection(self, description):
        self.gdfignition = self.gdfignitionAll[self.gdfignitionAll['description'] == description]
        self.dflandscape = self.dflandscapeAll[self.dflandscapeAll['description'] == description]
        
        
    def create_rundir(self):
        return self.fp.create_rundir()
    
    def append(self, data: dict):
        filetype = data['filetype']
        
        if filetype == 'Simulation':
            # Read the output simulation geoms
            if os.path.exists(data['filepath']):
                gdf = gpd.read_file(data['filepath'])
                idxlst = []
                geomlst = []
                igniteidxlst = []
                compareidxlst = []
                descriptionlst = []
                datetimelst = []
                filepathlst = []
                windspeedlst = []
                winddirectionlst = []
                configpathlst = []

                # For each elapsed time
                minuteslst = gdf['Elapsed_Mi'].unique()

                for minutespassed in minuteslst:
                    gdf0 = gdf[gdf['Elapsed_Mi'] == minutespassed]
                    polygon_lst = [Polygon(value) for value in gdf0['geometry'].values]
                    multipoly = MultiPolygon()
                    for poly in polygon_lst:
                        multipoly = multipoly.union(poly.buffer(0))
                    geomlst.append(multipoly)

                    # unique id
                    uniqueid = uuid.uuid4().hex
                    idxlst.append(uniqueid)

                    igniteidxlst.append(data['igniteidx'])
                    compareidxlst.append(data['compareidx'])
                    descriptionlst.append(data['description'])
                    datetimelst.append(data['startdt'] + datetime.timedelta(minutes=minutespassed))
                    filepathlst.append(data['filepath'])
                    windspeedlst.append(data['windspeed'])
                    winddirectionlst.append(data['winddirection'])
                    configpathlst.append(data['configpath'])

                # Create the gdf for appending
                gdfappend = gpd.GeoDataFrame({'igniteidx': igniteidxlst,
                                              'compareidx': compareidxlst,
                                              'description': descriptionlst,
                                              'datetime': datetimelst,
                                              'filepath': filepathlst,
                                              'windspeed': windspeedlst,
                                              'winddirection': winddirectionlst,
                                              'configpath': configpathlst},
                                         geometry=geomlst,
                                         index=idxlst,
                                         crs=self.gdfignition.crs)
                self.gdfsimulation = self.gdfsimulation.append(gdfappend)
        else:
            print(f'filetype = {filetype} not yet implemented!')
            
    def startdt(self, igniteidx):
        return self.gdfignition.loc[igniteidx, 'datetime']
    
    def lcppath(self, lcpidx):
        return self.dflandscape.loc[lcpidx, 'filepath']
    def ignitepath(self, igniteidx):
        return self.gdfignition.loc[igniteidx, 'filepath']
    def barrierpath(self, barrieridx):
        return self.dfbarrier.loc[barrieridx, 'filepath']
    
class User:
    def __init__(self, fp: FilePaths):
        # Object to keep the farsite files organized
        self.fp = fp
        
        # Setup the main components
        self.__setup()
        
    def __setup(self):
        # setup the database for create/append
        self.__setup_dbtable()
        
        # Setup the interface for data input
        self.__setup_interface()
        
    def __setup_interface(self):
        print('Setting up the interface')

        # Windspeed
        windspeed_desc = Label('Windspeed')
        windspeed_widget = IntRangeSlider(min=0, max=50, value=(0,20))
        windspeed_units = Label('mph')

        windspeedstep_desc = Label('Steps')
        windspeedstep_widget = IntSlider(min=1, max=windspeed_widget.max)

        windspeed_box = HBox([windspeed_desc, windspeed_widget, windspeedstep_desc, windspeedstep_widget, windspeed_units])

        # Winddirection
        winddirection_desc = Label('Winddirection')
        winddirection_widget = IntRangeSlider(min=0, max=355, step=5, value=(0,180))
        winddirection_units = Label('degrees')

        winddirectionstep_desc = Label('Steps')
        winddirectionstep_widget = IntSlider(min=5, max=355, step=5)

        winddirection_box = HBox([winddirection_desc, winddirection_widget, winddirectionstep_desc, winddirectionstep_widget, winddirection_units])

        # Temperature
        temperature_desc = Label('Temperature')
        temperature_widget = IntSlider(min=-40, max=120, value=60)
        temperature_units = Label('Fahrenheit')

        temperature_box = HBox([temperature_desc, temperature_widget, temperature_units])

        # Relative humidity
        relhumid_desc = Label('Relative Humidity')
        relhumid_widget = IntSlider(min=0, max=100, value=10)
        relhumid_units = Label('%')

        relhumid_box = HBox([relhumid_desc, relhumid_widget, relhumid_units])

        # Burn time
        burntime_desc = Label('Burn time')
        burntime_widget = IntSlider(min=30, max=300, value=3, step=30)
        burntime_units = Label('minutes')

        burntimestep_desc = Label('Steps')
        burntimestep_widget = IntSlider(min=30, max=300, step=30)

        burntime_box = HBox([burntime_desc, burntime_widget, burntimestep_desc, burntimestep_widget, burntime_units])

        # Calculate button
        calculate_widget = Button(description='Calculate Perimeters')

        def setWinddirectionStepMax(change):
            if change['name'] == 'value':
                winddirectionstep_widget.max = winddirection_widget.value[1] - winddirection_widget.value[0]
        def setWindspeedStepMax(change):
            if change['name'] == 'value':
                windspeedstep_widget.max = windspeed_widget.value[1] - windspeed_widget.value[0]
        def setBurntimeStepMax(change):
            if change['name'] == 'value':
                burntimestep_widget.max = burntime_widget.value

        def calculateClicked(change):
            windspeedlow = windspeed_widget.value[0]
            windspeedhigh = windspeed_widget.value[1]
            windspeeddelta = windspeedstep_widget.value

            winddirectionlow = winddirection_widget.value[0]
            winddirectionhigh = winddirection_widget.value[1]
            winddirectiondelta = winddirectionstep_widget.value

            relhumid = relhumid_widget.value
            burntime = datetime.timedelta(minutes=burntime_widget.value)
            burntimestep = datetime.timedelta(minutes=burntimestep_widget.value)

            temperature = temperature_widget.value
            
            inputData = {'windspeedlow': windspeedlow, 'windspeedhigh': windspeedhigh, 'windspeeddelta': windspeeddelta,
                         'winddirectionlow': winddirectionlow, 'winddirectionhigh': winddirectionhigh, 'winddirectiondelta': winddirectiondelta, 
                         'relhumid': relhumid, 'burntime': burntime, 'burntimestep': burntimestep, 'temperature': temperature}

            self.mainapi = self.calculatePerimeters(inputData)
            

        # Collect activity
        windspeed_widget.observe(setWindspeedStepMax)
        burntime_widget.observe(setBurntimeStepMax)
        calculate_widget.on_click(calculateClicked)

############################################################################               
#         # Loading widget
#         self.loading_widget = IntProgress(
#             value=0,
#             min=0,
#             max=100,
#             description='Calculating:',
#             style={'bar_color': '#0000FF'},
#             orientation='horizontal'
#         )

#         value_label = Label(str(self.loading_widget.value) + ' %')

#         def update_label(vlabel, *args):
#             vlabel.value = str(args[0]['new']) + ' %'

#         self.loading_widget.observe(partial(update_label, value_label))

#         loading_box = HBox([self.loading_widget, value_label])
############################################################################       

        
        ###### Choose ignite and compare perimeters
        # Dependencies
        def initiate_perimeters(gdf, newigniteid, newcompareid):
            add_mask_ignite = (gdf['objectid'] == newigniteid)
            add_mask_compare = (gdf['objectid'] == newcompareid)

            gdf['WKTLayerIgnite'] = gdf.apply(lambda row: WKTLayer(wkt_string = row['geometry'].wkt), axis=1)
            gdf['WKTLayerCompare'] = gdf.apply(lambda row: WKTLayer(wkt_string = row['geometry'].wkt), axis=1)
            
            for layer in m.layers:
                if isinstance(layer, WKTLayer):
                    m.remove_layer(layer)
                    
            gdf.loc[add_mask_ignite, 'WKTLayerIgnite'].apply(lambda wlayer: m.add_layer(wlayer))
            gdf.loc[add_mask_compare, 'WKTLayerCompare'].apply(lambda wlayer: m.add_layer(wlayer))


        def update_perimeters(gdf, vbox, oldigniteid, newigniteid, oldcompareid, newcompareid):    
            remove_mask_ignite = (gdf['objectid'] == oldigniteid)
            remove_mask_compare = (gdf['objectid'] == oldcompareid)
            add_mask_ignite = (gdf['objectid'] == newigniteid)
            add_mask_compare = (gdf['objectid'] == newcompareid)

            gdf.loc[remove_mask_ignite, 'WKTLayerIgnite'].apply(lambda wlayer: m.remove_layer(wlayer))
            gdf.loc[add_mask_ignite, 'WKTLayerIgnite'].apply(lambda wlayer: m.add_layer(wlayer))
            gdf.loc[remove_mask_compare, 'WKTLayerCompare'].apply(lambda wlayer: m.remove_layer(wlayer))
            gdf.loc[add_mask_compare, 'WKTLayerCompare'].apply(lambda wlayer: m.add_layer(wlayer))

            vbox.children[1].value = str(gdf.set_index('objectid').loc[newigniteid, 'datetime'])
            vbox.children[3].value = str(gdf.set_index('objectid').loc[newcompareid, 'datetime'])

            deltadt = gdf.set_index('objectid').loc[newcompareid, 'datetime'] - gdf.set_index('objectid').loc[newigniteid, 'datetime']
            vbox.children[4].value = str(int(deltadt.total_seconds()/60)) + ' minutes'

        def observe_objectid_slider(m, gdf, vbox, event):
            if event['owner'].options.count(event['old']) == 0: 
                return None
            
            self.__selectTimeParams()

            if event['owner'].description == 'ignite':
                oldigniteid = event['old']
                newigniteid = event['new']
                oldcompareid = vbox.children[2].value
                newcompareid = vbox.children[2].value
            elif event['owner'].description == 'compare':
                oldigniteid = vbox.children[0].value
                newigniteid = vbox.children[0].value
                oldcompareid = event['old']
                newcompareid = event['new']

            update_perimeters(gdf, vbox, oldigniteid, newigniteid, oldcompareid, newcompareid)
            
        
        def observe_fire_dropdown(m, igniteid_select, compareid_select, vbox, event):
            description = event['new']
            self.db.filter_selection(description)
            print(f'filtering selection is done for {description}')
            compareid_select.options = self.db.gdfignition['objectid']
            print(f'Compareid options are recalculated')
            igniteid_select.options = self.db.gdfignition['objectid']
            print(f'igniteid options are recalculated')
            
            
            m.center = (self.db.gdfignition.iloc[0]['geometry'].centroid.y, 
                        self.db.gdfignition.iloc[0]['geometry'].centroid.x)
            print(f'Map is recentered')
            
            igniteid_select.unobserve(self.objectid_slider_handle, names='value')
            compareid_select.unobserve(self.objectid_slider_handle, names='value')
            gdf = self.db.gdfignition
            initiate_perimeters(gdf, gdf['objectid'].iloc[0], gdf['objectid'].iloc[0])
            
            self.objectid_slider_handle = partial(observe_objectid_slider, m, gdf, vbox)
            igniteid_select.observe(self.objectid_slider_handle, names='value')
            compareid_select.observe(self.objectid_slider_handle, names='value')
            
            print(f'Perimeters are recalculated')
            


        centerlat = 34.178861487501464
        centerlon = -118.566380281569

        m = Map(
            basemap=basemaps.Esri.WorldTopoMap,
            center=(centerlat, centerlon),
            zoom=10,
            layout=Layout(height='800px', width='100%'),
            zoom_control=False
        )

        gdf = self.db.gdfignition.to_crs(epsg=4326)
        
        objectids = gdf['objectid'].unique()

        igniteid_select = SelectionSlider(description='ignite', options=objectids)
        ignitedt_label = Label(value=str(gdf.set_index('objectid').loc[igniteid_select.value, 'datetime']))

        compareid_select = SelectionSlider(description='compare', options=objectids)
        comparedt_label = Label(value=str(gdf.set_index('objectid').loc[compareid_select.value, 'datetime']))

        deltadt = gdf.set_index('objectid').loc[compareid_select.value, 'datetime'] - gdf.set_index('objectid').loc[igniteid_select.value, 'datetime']
        deltadt_label = Label(value=str(int(deltadt.total_seconds()/60)) + ' minutes')

        fire_select = Dropdown(options=self.db.gdfignitionAll['description'].unique())
        
        initiate_perimeters(gdf, igniteid_select.value, compareid_select.value)

        vbox = VBox([igniteid_select, ignitedt_label, compareid_select, comparedt_label, deltadt_label, fire_select])

        self.objectid_slider_handle = partial(observe_objectid_slider, m, gdf, vbox)
        
        igniteid_select.observe(self.objectid_slider_handle, names='value')
        compareid_select.observe(self.objectid_slider_handle, names='value')
        
        self.fire_dropdown_handle = partial(observe_fire_dropdown, m, igniteid_select, compareid_select, vbox)
        fire_select.observe(self.fire_dropdown_handle, names='value')

        igniteid_box = HBox([igniteid_select, ignitedt_label, fire_select])
        compareid_box = HBox([compareid_select, comparedt_label, deltadt_label])
        ## Combine all boxes
        
        self.UI = VBox([igniteid_box, compareid_box, windspeed_box, winddirection_box,  burntime_box, temperature_box, relhumid_box, calculate_widget])
        
        widget_control = WidgetControl(widget = self.UI, position='topright')
        
        m.add_control(ZoomControl(position='topleft'))
        m.add_control(ScaleControl(position='topleft'))

        m.add_control(widget_control)
        
        self.m = m
    
    def __setup_dbtable(self):
        print('Database interaction not yet implemented. Use pickle file for dataframes instead!')
        
        self.db = Database(self.fp)
            
    def __selectPerimeter(self):
        # Choose a perimeter from the database
        print('Choosing a perimeter from the database')
#         self.igniteidx = '9f82e870591748a9a8a01346d174f2a1'
        
        igniteid = self.UI.children[0].children[0].value
        self.igniteidx = self.db.gdfignition.reset_index().set_index('objectid').loc[igniteid, 'index']
        
        compareid = self.UI.children[1].children[0].value
        self.compareidx = self.db.gdfignition.reset_index().set_index('objectid').loc[compareid, 'index']
        # Grab the landscape from gdal
        print('Collecting lcp file from gdal_translate')
        
        # Append the lcp to the database (Check for existence?)
        self.lcpidx = self.db.dflandscape.index[0] # Maria fire
        
        # Select the barrier
        self.barrieridx = 'cb47616cd2dc4ccc8fd523bd3a5064bb' # No Barrier
        
    def __selectWindParams(self, inputData: dict):
        # Choose a windspeed range
        self.windspeedlow = inputData['windspeedlow']
        self.windspeedhigh = inputData['windspeedhigh']
        self.windspeeddelta = inputData['windspeeddelta']
        
        # Choose a winddirection range
        self.winddirectionlow = inputData['winddirectionlow']
        self.winddirectionhigh = inputData['winddirectionhigh']
        self.winddirectiondelta = inputData['winddirectiondelta']
        
    def __selectTimeParams(self):
        # Ignition is read from the dftable
        igniteid = self.UI.children[0].children[0].value
        self.startdt = self.db.gdfignition.reset_index().set_index('objectid').loc[igniteid, 'datetime']
        
#         self.deltadt = inputData['burntimestep']

#         self.enddt = self.startdt + inputData['burntime']

        compareid = self.UI.children[1].children[0].value
        self.enddt = self.db.gdfignition.reset_index().set_index('objectid').loc[compareid, 'datetime']
        
        self.deltadt = self.enddt - self.startdt
        
        burnwidget = self.UI.children[4].children[1]
        burnwidget.value = str(int(self.deltadt.total_seconds()/60))
        burnwidget.disabled = True
        
        burnstepwidget = self.UI.children[4].children[3]
        burnstepwidget.value = burnwidget.value
        burnstepwidget.disabled = True
        
    def __selectHumidity(self, inputData: dict):
        self.humidity = inputData['relhumid']
        
    def __selectTemperature(self, inputData: dict):
        self.temperature = inputData['temperature']
    
    def __selectInputParams(self, inputData: dict):
        # Select ignition perimeter
        self.__selectPerimeter() # Automatically sets the landcape
        
        # Choose windspeed range
        self.__selectWindParams(inputData)
        
        # Choose time parameters
        self.__selectTimeParams()
        
        # Choose humidity
        self.__selectHumidity(inputData)
        
        # Choose temperature
        self.__selectTemperature(inputData)
        
        
        windspeed_range = range(self.windspeedlow, self.windspeedhigh, self.windspeeddelta)
        winddirection_range = range(self.winddirectionlow, self.winddirectionhigh, self.winddirectiondelta)
        
        # Remaining params are default at the moment, and automatically set in the ConfigFile
        self.inputData = Input(startdt = self.startdt, enddt = self.enddt, deltadt = self.deltadt,
                               igniteidx = self.igniteidx, compareidx = self.compareidx, 
                               description = self.fire_dropdown_handle.args[3].children[5].value,
                               lcpidx = self.lcpidx, barrieridx = self.barrieridx,
                               windspeed_lst = list(windspeed_range), winddirection_lst = list(winddirection_range),
                               humidity = self.humidity, temperature = self.temperature)


    def calculatePerimeters(self, inputData):
        print(inputData)
        
        # Select all the parameters
        self.__selectInputParams(inputData)
        
        # Call the main API
        mainAPI = Main(inputData = self.inputData, 
                       db = self.db)
        
        return mainAPI

class Main:
    def __init__(self, inputData : Input, db: Database):
        
        # Set the input
        self.inputData = inputData
        
        # Set the filepaths
#         self.fp = fp
        
        # Database
#         self.dftable = dftable
        self.db = db
        
        self.runfile_lst = []
        self.runfile_done = {}
        
        windspeed_lst = [ 1,  1,  1,  0,  1,  1,  1,  1,  3,  1,  2,  3,  2,  2,  2,  2,  3, 3,  3,  5,  4,  5,  6,  6,  8,  9, 10]
        winddirection_lst = [315,  90,  90, 180, 270, 117, 135, 180, 259, 207, 236, 243, 198, 214, 225, 194, 207, 217, 202, 234, 214, 216, 217, 221, 197, 196, 193]
        
#         for ws in self.inputData.windspeed_lst:
#             for wd in self.inputData.winddirection_lst:
        for ws, wd in zip(windspeed_lst, winddirection_lst):
            # Set Run file
            runfile = Run_File(mainapi = self, db = self.db, windspeed = ws, winddirection = wd, 
                               startdt = self.inputData.startdt, enddt = self.inputData.enddt, deltadt = self.inputData.deltadt,
                               lcpidx = self.inputData.lcpidx, 
                               igniteidx = self.inputData.igniteidx,
                               compareidx = self.inputData.compareidx,
                               description = self.inputData.description,
                               barrieridx = self.inputData.barrieridx,
                               temperature = self.inputData.temperature,
                               humidity = self.inputData.humidity)

            # Create runfile and configfiles
            self.runfile_lst.append(runfile)
            self.runfile_done[runfile] = 0
                
        self.__setup_farsite()
        
    def __setup_farsite(self):
        self.farsite_lst = []
        for runfile in self.runfile_lst:
            self.farsite_lst.append(FarsiteManual(runfile))
    
    def run_farsite(self, numproc=1):
        
        if numproc == 1:
            for farsite in self.farsite_lst:
                farsite.updatedb(farsite.run_command())
        else:
            pool = Pool(processes=numproc)

            # Run for each FarsiteManual
            for farsite in self.farsite_lst:
                pool.apply_async(farsite.run_command, callback=farsite.updatedb)

            pool.close()
            pool.join()

        
############################################################################       
#     def update_loading(self):
#         # Count the number of runfiles done
#         count = 0.0
#         for (runfile, value) in self.runfile_done.items():
#             count += value
        
#         self.loading_widget.value = int(count/len(self.runfile_lst)*100)
############################################################################       

class Run_File:
    def __init__(self, mainapi: Main, db: Database, windspeed: int, winddirection: int,
                 startdt: datetime.datetime, enddt: datetime.datetime, deltadt: datetime.timedelta,
                 lcpidx: str, igniteidx: str, compareidx: str, description: str, barrieridx: str,
                 temperature: int, humidity: int):
        
        # Setup the parameters
        self.mainapi = mainapi
        self.windspeed = windspeed
        self.winddirection = winddirection
        self.startdt = startdt
        self.enddt = enddt
        self.deltadt = deltadt
        self.db = db
        self.temperature = temperature
        self.humidity = humidity
        
        # Set Config file
        self.configfile = Config_File(FARSITE_START_TIME = self.startdt,
                                 FARSITE_END_TIME = self.enddt,
                                 windspeed = self.windspeed,
                                 winddirection = self.winddirection)
        
        # Set additional information
        self.configfile.FARSITE_TIMESTEP = int(self.deltadt.total_seconds()/60)
        self.configfile.temperature = self.temperature
        self.configfile.humidity = self.humidity
        
        # Directory to keep the input files
        rundir = db.create_rundir()
        
        # Input filepaths
        self.configpath = os.path.join(rundir, 'config')
        self.runpath = os.path.join(rundir, 'run')        
        self.outpath = os.path.join(rundir, 'out')
        
        # Read the remaining filepaths
        self.lcppath = self.db.lcppath(lcpidx)
        self.ignitepath = self.db.ignitepath(igniteidx)
        self.barrierpath = self.db.barrierpath(barrieridx)
        
        # Keep a record of igniteidx to update the table with simulation data
        self.igniteidx = igniteidx
        self.compareidx = compareidx
        self.description = description

    def tostring(self):
        return '{lcpath} {cfgpath} {ignitepath} {barrierpath} {outpath} -1'.format(
                                lcpath =  self.lcppath, 
                                cfgpath = self.configpath, 
                                ignitepath = self.ignitepath, 
                                barrierpath = self.barrierpath, 
                                outpath = self.outpath)
    
    def tofile(self):
        # Write Runfile
#         print('Writing to runfile')
        with open(self.runpath, mode='w') as file:
            file.write(self.tostring())
#         print('Writing to configfile')
        # Write configfile
        with open(self.configpath, mode='w') as file:
            file.write(self.configfile.tostring())
#         print('Writing is done')
            
    def updatedb(self):
        data = {'filetype': 'Simulation',
                'igniteidx': self.igniteidx,
                'compareidx': self.compareidx,
                'description': self.description,
                'startdt': self.startdt,
                'filepath': self.outpath + '_Perimeters.shp',
                'windspeed': self.windspeed,
                'winddirection': self.winddirection,
                'configpath': self.configpath}
                
        self.db.append(data)
        
        # Calculation done for runfile
        self.mainapi.runfile_done[self] = 1
        
        # Update the loading widget
############################################################################       
#         self.mainapi.update_loading()
        
#         self.dftable.loc[uniqueid, 'filetype'] = 'Simulation'
#         self.dftable.loc[uniqueid, 'igniteidx'] = self.igniteidx
#         self.dftable.loc[uniqueid, 'datetime'] = self.startdt
#         self.dftable.loc[uniqueid, 'filepath'] = self.outpath + '_Perimeters.shp'
            
#         return 0
############################################################################       
#TODO: Read all the values in the params and create the config file accordingly
# This is a config file parser

class Config_File:
    def __init__(self, 
                 FARSITE_START_TIME: datetime, 
                 FARSITE_END_TIME: datetime, 
                 windspeed: float, winddirection: float):
        self.__set_default()
        
        # Set the parameters
        self.FARSITE_TIMESTEP = int((FARSITE_END_TIME - FARSITE_START_TIME).total_seconds()/60)
        self.FARSITE_START_TIME = datetime.datetime(2019, 9, 9, 19, 0)
        self.FARSITE_END_TIME = self.FARSITE_START_TIME + (FARSITE_END_TIME - FARSITE_START_TIME)
        self.windspeed = windspeed
        self.winddirection = winddirection

    def __set_default(self):
        self.version = 1.0
        self.FARSITE_DISTANCE_RES = 30
        self.FARSITE_PERIMETER_RES = 60
        self.FARSITE_MIN_IGNITION_VERTEX_DISTANCE = 15.0
        self.FARSITE_SPOT_GRID_RESOLUTION = 60.0
        self.FARSITE_SPOT_PROBABILITY = 0.9
        self.FARSITE_SPOT_IGNITION_DELAY = 0
        self.FARSITE_MINIMUM_SPOT_DISTANCE = 60
        self.FARSITE_ACCELERATION_ON = 1
        self.FARSITE_FILL_BARRIERS = 1
        self.SPOTTING_SEED = 253114
        
        self.FUEL_MOISTURES_DATA = [[0, 3, 4, 6, 30, 60]]
        
        self.RAWS_ELEVATION = 2501
        self.RAWS_UNITS = 'English'
        # Add self.raws from the init
              
        self.FOLIAR_MOISTURE_CONTENT = 100
        self.CROWN_FIRE_METHOD = 'ScottReinhardt'
        
        self.WRITE_OUTPUTS_EACH_TIMESTEP = 0
        
        self.temperature = 66
        self.humidity = 8
        self.precipitation = 0
        self.cloudcover = 0
        
    def tostring(self):
        config_text = 'FARSITE INPUTS FILE VERSION {}\n'.format(self.version)
        
        str_start = '{month} {day} {time}'.format(
                            month = self.FARSITE_START_TIME.month,
                            day = self.FARSITE_START_TIME.day,
                            time = '{}{:02d}'.format(
                                    self.FARSITE_START_TIME.hour,
                                    self.FARSITE_START_TIME.minute))
        config_text += 'FARSITE_START_TIME: {}\n'.format(str_start)

        str_end = '{month} {day} {time}'.format(
                            month = self.FARSITE_END_TIME.month,
                            day = self.FARSITE_END_TIME.day,
                            time = '{}{:02d}'.format(
                                    self.FARSITE_END_TIME.hour,
                                    self.FARSITE_END_TIME.minute))
        config_text += 'FARSITE_END_TIME: {}\n'.format(str_end)
        
        config_text += 'FARSITE_TIMESTEP: {}\n'.format(self.FARSITE_TIMESTEP)
        config_text += 'FARSITE_DISTANCE_RES: {}\n'.format(self.FARSITE_DISTANCE_RES)
        config_text += 'FARSITE_PERIMETER_RES: {}\n'.format(self.FARSITE_PERIMETER_RES)
        config_text += 'FARSITE_MIN_IGNITION_VERTEX_DISTANCE: {}\n'.format(self.FARSITE_MIN_IGNITION_VERTEX_DISTANCE)
        config_text += 'FARSITE_SPOT_GRID_RESOLUTION: {}\n'.format(self.FARSITE_SPOT_GRID_RESOLUTION)
        config_text += 'FARSITE_SPOT_PROBABILITY: {}\n'.format(self.FARSITE_SPOT_PROBABILITY)
        config_text += 'FARSITE_SPOT_IGNITION_DELAY: {}\n'.format(self.FARSITE_SPOT_IGNITION_DELAY)
        config_text += 'FARSITE_MINIMUM_SPOT_DISTANCE: {}\n'.format(self.FARSITE_MINIMUM_SPOT_DISTANCE)
        config_text += 'FARSITE_ACCELERATION_ON: {}\n'.format(self.FARSITE_ACCELERATION_ON)
        config_text += 'FARSITE_FILL_BARRIERS: {}\n'.format(self.FARSITE_FILL_BARRIERS)
        config_text += 'SPOTTING_SEED: {}\n'.format(self.SPOTTING_SEED)
        
        # Fuel moistures
        config_text += 'FUEL_MOISTURES_DATA: {}\n'.format(len(self.FUEL_MOISTURES_DATA))
        for data in self.FUEL_MOISTURES_DATA:
            config_text += '{} {} {} {} {} {}\n'.format(data[0], data[1], data[2],
                                                      data[3], data[4], data[5])
            
        config_text += 'RAWS_ELEVATION: {}\n'.format(self.RAWS_ELEVATION)
        config_text += 'RAWS_UNITS: {}\n'.format(self.RAWS_UNITS)
        
        # Weather data (currently only a single weather data)
        config_text += 'RAWS: 1\n'
        config_text += '{year} {month} {day} {time} {temperature} {humidity} {precipitation} {windspeed} {winddirection} {cloudcover}\n'.format(
                                year = self.FARSITE_START_TIME.year,
                                month = self.FARSITE_START_TIME.month,
                                day = self.FARSITE_START_TIME.day,
                                time = '{}{:02d}'.format(
                                    self.FARSITE_START_TIME.hour, 
                                    self.FARSITE_START_TIME.minute),
                                temperature = self.temperature,
                                humidity = self.humidity,
                                precipitation = self.precipitation,
                                windspeed = self.windspeed,
                                winddirection = self.winddirection,
                                cloudcover = self.cloudcover
                            )
        config_text += 'FOLIAR_MOISTURE_CONTENT: {}\n'.format(self.FOLIAR_MOISTURE_CONTENT)
        config_text += 'CROWN_FIRE_METHOD: {}\n'.format(self.CROWN_FIRE_METHOD)
        config_text += 'WRITE_OUTPUTS_EACH_TIMESTEP: {}'.format(self.WRITE_OUTPUTS_EACH_TIMESTEP)
        
        return config_text
    
class FarsiteManual:
    def __init__(self, runfile: Run_File, farsitepath = '/home/tcaglar/farsite/TestFARSITE', timeout: int = 5):
        # Setup farsite manual api
        self.farsitepath = farsitepath
        self.timeout = timeout
        
        self.runfile = runfile # Create the input files
        
        self.__setup_command()
    def __setup_command(self, ncores=4):
        # Timeout 1 minute.
        self.command = f'timeout {self.timeout}m {self.farsitepath} {self.runfile.runpath} {ncores}'  # donot run in background
        
    def run_command(self):
        # TODO 
        # Create the folder defined by the runfile
#         print(self.command)
        # Write into the folder
        self.runfile.tofile()
#         print('Runfile written')
        
        # Run the command in os
        os.system(self.command)

#         self.updatedb()
#         # Return
        return 0
    def updatedb(self, value):
#         print(value)
        self.runfile.updatedb()