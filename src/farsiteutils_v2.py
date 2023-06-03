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
        
    windspeed: int
    winddirection: int
        
    temperature: int
    humidity: int

class FilePaths:
    def __init__(self, datadir):
        self.datadir = datadir
        self.dfpath = os.path.join(self.datadir, 'dftable_06032023.pkl')
    
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

def change_username_jovyan(df, column):
    for ix, row in df.iterrows():
        path_list = row[column].split('/')
        path_list[2] = 'jovyan'

        path = ''
        for string in path_list[:-1]:
            path += f'{string}/'
        path += path_list[-1]

        df.loc[ix, column] = path        
        
class Database:
    def __init__(self, fp: FilePaths):
        # Setup params
        self.fp = fp
        
        # TODO
        # Setup the database for reading
        
        
        try:
            dftable = pd.read_pickle(self.fp.dfpath)
            change_username_jovyan(dftable, 'filepath')
        except FileNotFoundError:
            print(f'\n!!Caution!! Path {self.fp.dfpath} not found! Cannot choose ignition!!\n')
            raise
            
        # Collect the tables in dataframe format
        # Table 1 - ignition
        self.gdfignitionAll = gpd.GeoDataFrame(dftable[dftable['filetype'] == 'Ignition'])
        for (idx, ignition) in self.gdfignitionAll.iterrows():
            geom = gpd.read_file(ignition['filepath']).loc[0,'geometry']
            self.gdfignitionAll.loc[idx, 'shape'] = gpd.GeoSeries(geom).to_wkb().iloc[0]

        gs = gpd.GeoSeries.from_wkb(self.gdfignitionAll['shape'])
        self.gdfignitionAll['geometry'] = gs
        self.gdfignitionAll = self.gdfignitionAll.drop(columns='shape').set_crs(epsg=5070)
        
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
                self.gdfsimulation = pd.concat([self.gdfsimulation,
                                                gdfappend])
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

    def __setup_dbtable(self):
        print('Database interaction not yet implemented. Use pickle file for dataframes instead!')
        
        self.db = Database(self.fp)
            
    def __selectPerimeter(self, inputData: dict):
        # Choose a perimeter from the database
        print('Choosing a perimeter from the database')
        self.igniteidx = inputData['igniteidx']
        self.compareidx = inputData['compareidx']
        self.lcpidx = inputData['lcpidx']
        self.barrieridx = inputData['barrieridx']
        
        self.description = inputData['description']

    def __selectWindParams(self, inputData: dict):
        self.windspeed = inputData['windspeed']
        self.winddirection = inputData['winddirection']
        
    def __selectTimeParams(self):
        # Ignition is read from the dftable
        self.startdt = self.db.gdfignition.loc[self.igniteidx, 'datetime']
        self.enddt = self.db.gdfignition.loc[self.compareidx, 'datetime']
        self.deltadt = self.enddt - self.startdt
        
    def __selectHumidity(self, inputData: dict):
        self.humidity = inputData['relhumid']
        
    def __selectTemperature(self, inputData: dict):
        self.temperature = inputData['temperature']
    
    def __selectInputParams(self, inputData: dict):
        # Select ignition perimeter
        self.__selectPerimeter(inputData) # Automatically sets the landcape
        
        # Choose windspeed range
        self.__selectWindParams(inputData)
        
        # Choose time parameters
        self.__selectTimeParams()
        
        # Choose humidity
        self.__selectHumidity(inputData)
        
        # Choose temperature
        self.__selectTemperature(inputData)
        
        # Remaining params are default at the moment, and automatically set in the ConfigFile
        self.inputData = Input(startdt = self.startdt, enddt = self.enddt, deltadt = self.deltadt,
                               igniteidx = self.igniteidx, compareidx = self.compareidx, 
                               description = self.description,
                               lcpidx = self.lcpidx, barrieridx = self.barrieridx,
                               windspeed = self.windspeed, winddirection = self.winddirection,
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
        
        # Database
        self.db = db
        
        self.runfile_lst = []
        self.runfile_done = {}
                
        # Set Run file
        runfile = Run_File(mainapi = self, db = self.db, 
                           windspeed = self.inputData.windspeed, winddirection = self.inputData.winddirection, 
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
            self.farsite_lst.append(Farsite(runfile))
    
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
        with open(self.runpath, mode='w') as file:
            file.write(self.tostring())
            
        # Write configfile
        with open(self.configpath, mode='w') as file:
            file.write(self.configfile.tostring())
            
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

class Config_File:
    def __init__(self, 
                 FARSITE_START_TIME: datetime, 
                 FARSITE_END_TIME: datetime, 
                 windspeed: int, winddirection: int):
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
        self.FARSITE_SPOT_PROBABILITY = 0  # 0.9
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
    
class Farsite:
    def __init__(self, runfile: Run_File, farsitepath = '/home/jovyan/farsite/TestFARSITE', timeout: int = 5):
        # Setup farsite manual api
        self.farsitepath = farsitepath
        self.timeout = timeout   # in minutes
        
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