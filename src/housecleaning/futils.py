import datetime
import os
import pathlib
import uuid

from shapely import Polygon, make_valid, GeometryCollection, MultiPolygon
from pyproj import Transformer
from putils import calculate_max_area_geom, validate_geom

import geopandas as gpd

import warnings

class Config_File:
    def __init__(self, 
                 FARSITE_START_TIME: datetime.datetime, 
                 FARSITE_END_TIME: datetime.datetime,
                 windspeed: int, winddirection: int,
                 FARSITE_DISTANCE_RES: int,
                 FARSITE_PERIMETER_RES: int):
        self.__set_default()
        
        # Set the parameters
        self.FARSITE_TIMESTEP = int((FARSITE_END_TIME - FARSITE_START_TIME).total_seconds()/60)
        self.FARSITE_START_TIME = datetime.datetime(2019, 9, 9, 19, 0)
        self.FARSITE_END_TIME = self.FARSITE_START_TIME + (FARSITE_END_TIME - FARSITE_START_TIME)
        self.FARSITE_DISTANCE_RES = FARSITE_DISTANCE_RES
        self.FARSITE_PERIMETER_RES = FARSITE_PERIMETER_RES
        self.windspeed = windspeed
        self.winddirection = winddirection

    def __set_default(self):
        self.version = 1.0
        self.FARSITE_DISTANCE_RES = 60
        self.FARSITE_PERIMETER_RES = 120
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
                            time = '{:02d}{:02d}'.format(
                                    self.FARSITE_START_TIME.hour,
                                    self.FARSITE_START_TIME.minute))
        config_text += 'FARSITE_START_TIME: {}\n'.format(str_start)

        str_end = '{month} {day} {time}'.format(
                            month = self.FARSITE_END_TIME.month,
                            day = self.FARSITE_END_TIME.day,
                            time = '{:02d}{:02d}'.format(
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
                                time = '{:02d}{:02d}'.format(
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
    
    def to_file(self, filepath: str):
        with open(filepath, mode='w') as file:
            file.write(self.tostring())
            
            
class Run_File:
    def __init__(self, lcppath: str, cfgpath: str, ignitepath: str, barrierpath: str, outpath: str):
        self.lcppath = lcppath
        self.cfgpath = cfgpath
        self.ignitepath = ignitepath
        self.barrierpath = barrierpath
        self.outpath = outpath

    def tostring(self):
        return '{lcpath} {cfgpath} {ignitepath} {barrierpath} {outpath} -1'.format(
                                lcpath =  self.lcppath, 
                                cfgpath = self.cfgpath, 
                                ignitepath = self.ignitepath, 
                                barrierpath = self.barrierpath, 
                                outpath = self.outpath)
    def to_file(self, filepath: str):
        with open(filepath, mode='w') as file:
            file.write(self.tostring())

 
class Farsite:
    def __init__(self, initial: Polygon, params: dict, description: str, 
                 lcppath: str = None, barrierpath: str = None,
                 dist_res:int = 30, perim_res: int = 60,
                 debug:bool = False):
        
        self.farsitepath = os.path.join(os.getenv('HOME'), 'farsite-devAPI', 'TestFARSITE')        
        self.id = uuid.uuid4().hex

        self.tmpfolder = os.path.join(os.getenv('HOME'), 'farsite-devAPI', 'data', 'tmp',
                                      datetime.datetime.today().strftime('%Y%m%d'))
        pathlib.Path(self.tmpfolder).mkdir(parents=True, exist_ok=True) 
        

        # Setup files
        self.description = description
        self.lcppath = lcppath + '.lcp'
        
        if lcppath == None:
            raise ValueErorr(f'The filepath {lcppath} cannot be used for lcppath')
            

        start_dt = datetime.datetime(year=2019, month=1, day=1, hour=10, minute=0)
        end_dt = start_dt + params['dt']
        windspeed = params['windspeed']
        winddirection = params['winddirection']
        
        #### RUN FILE PREPARATION ####
        # Config file
        self.config = Config_File(start_dt, end_dt, windspeed, winddirection, dist_res, perim_res)
        self.configpath = os.path.join(self.tmpfolder, f'{description}_config_{self.id}.cfg')
        self.config.to_file(self.configpath)
               
        # Barrier file
        self.barrierpath = barrierpath
        if self.barrierpath == None: # Use No Barrier
            self.barrierpath = os.path.join(os.getenv('HOME'), 'farsite-devAPI', 'inputs', 'barriers', 'NoBarrier', 'NoBarrier.shp')
            
        
        # Ignite path
        self.ignitepath = os.path.join(self.tmpfolder, f'{description}_ignite_{self.id}.shp')
        gpd.GeoDataFrame({'FID': [0], 'geometry': [initial]}).to_file(self.ignitepath)
        
        
        # Output path
        self.outpath = os.path.join(self.tmpfolder, f'{description}_out_{self.id}')
                                    
        
        # Generate RunFile
        self.runfile = Run_File(self.lcppath, self.configpath, self.ignitepath, self.barrierpath, self.outpath)
        self.runpath = os.path.join(self.tmpfolder, f'{description}_run_{self.id}')
        self.runfile.to_file(self.runpath)
        
        # Debugging keeps the files
        self.debug = debug

    def run(self, timeout=5, ncores=4):
        self.command = f'timeout {timeout}m {self.farsitepath} {self.runpath} {ncores} > output.out 2> output.err'  # donot run 
        os.system(self.command)
        
    def output_geom(self):
        output_path = self.outpath + '_Perimeters.shp'
        if not os.path.exists(output_path):
            return None
        
        gdf = gpd.read_file(output_path)
        if len(gdf) == 0:
            return None
        
        geom = gdf['geometry'][0]
        return Polygon(geom.coords)
        
    def __del__(self):
        if not self.debug:
            os.system(f'rm {os.path.join(self.tmpfolder, f"{self.description}_*_{self.id}*")}')
       

def generate_landscape(geom_5070: Polygon, description='test'):
    bounds = geom_5070.bounds

    ulx = bounds[0]-10000
    uly = bounds[3]+10000
    lrx = bounds[2]+10000
    lry=  bounds[1]-10000

    fname_lst = {'density': 'US_140CBD_12052016/Grid/us_140cbd', 
                 'base': 'US_140CBH_12052016/Grid/us_140cbh', 
                 'cover': 'US_140CC_12052016/Grid/us_140cc', 
                 'height': 'US_140CH_12052016/Grid/us_140ch', 
                 'fuel': 'US_140FBFM40_20180618/Grid/us_140fbfm40', 
                 'aspect': 'Aspect/Grid/us_asp', 
                 'elevation': 'DEM_Elevation/Grid/us_dem', 
                 'slope': 'Slope/Grid/us_slp'}
    type_lst = {'density': 'cbd',
                'base': 'cbh',
                'cover': 'cc',
                'height': 'ch',
                'fuel': 'fuel',   # fbfm40
                'aspect': 'aspect',
                'elevation': 'elevation', #dem
                'slope': 'slope'}

    from_folder = os.path.join('/data', 'firemap', 'landfire', 'mosaic')
    to_folder = os.path.join(os.getenv('HOME'), 'farsite-devAPI', 'inputs', 'landscapes')

    # Create the asc files
    ascpath_lst = {}
    for (key, fname) in fname_lst.items():
        ascpath_lst[key] = f'{os.path.join(to_folder, description)}-{key}.asc'
        
        run_script = f'gdal_translate -of AAIGrid -a_nodata -32768 -projwin {ulx} {uly} {lrx} {lry} {os.path.join(from_folder, fname)} {ascpath_lst[key]}'
        os.system(run_script)
        
    lcppath = os.path.join(to_folder, description)
    lcpmakepath = os.path.join(os.getenv('HOME'), 'farsite-devAPI', 'src', 'lcpmake')

    x,y = geom_5070.centroid.xy
    
    transformer = Transformer.from_crs(5070, 4326, always_xy=True)
    lat = transformer.transform(x[0], y[0])[1]
    
    base_command = f'{lcpmakepath} -latitude {lat} -landscape {lcppath}'
    run_command = base_command
    for (key, ascpath) in ascpath_lst.items():
        run_command += f' -{key} {ascpath}'

    os.system(run_command)
    
    # Remove unused files
    os.system(f"rm {os.path.join(to_folder, f'{description}*asc')} {os.path.join(to_folder, f'{description}*xml')} {os.path.join(to_folder, f'{description}*prj')}")
    
    
    # Copy 5070 projection to use it in farsite
    os.system(f"cp {os.path.join(to_folder, '5070.prj')} {os.path.join(to_folder, f'{description}.prj')}")
    return lcppath
            
    
def forward_pass_farsite(poly, params, lcppath, description, dist_res=30, perim_res=60, debug=False):
    MAX_SIM = 30 # minutes
    dt = params['dt']
    
    if dist_res > 500:
        warnings.warn(f'dist_res ({dist_res}) has to be 1-->500. Setting to 500')
        dist_res=500
    
    if perim_res > 500:
        warnings.warn(f'perim_res ({perim_res}) has to be 1-->500. Setting to 500')
        perim_res=500
    

    number_of_farsites = dt.seconds//(MAX_SIM*60)
    for i in range(number_of_farsites):
#         print(f'Calculating {i}/{number_of_farsites}')
        new_params = {'windspeed': params['windspeed'],
                      'winddirection': params['winddirection'],
                      'dt': datetime.timedelta(minutes=MAX_SIM)}
        farsite = Farsite(poly, new_params, lcppath=lcppath, description=description, dist_res=dist_res, perim_res=perim_res, debug=debug)
        farsite.run()
        if farsite.output_geom() is None:
            return None
        
        poly = validate_geom(farsite.output_geom())

    # Last step
    remaining_dt = dt - number_of_farsites*datetime.timedelta(minutes=MAX_SIM)
    if remaining_dt < datetime.timedelta(minutes=10):
#         print('Calculation is done')
        return poly
    
#     print('Calculating the last step')
    new_params = {'windspeed': params['windspeed'],
                  'winddirection': params['winddirection'],
                  'dt': remaining_dt}
    farsite = Farsite(poly, new_params, lcppath=lcppath, description=description)    
    farsite.run()
    
    return farsite.output_geom()