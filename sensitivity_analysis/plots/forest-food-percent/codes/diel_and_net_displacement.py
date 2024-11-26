# Import required libraries
#------------------------------------------------------------------------------------------------
import os
cwd = os.getcwd()
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity
import multiprocessing
import calendar
import pyproj
import pandas as pd
import numpy as np
from pyproj import Proj, transform
import matplotlib.pyplot as plt

#to supress warnings
import warnings     
warnings.filterwarnings('ignore') 

#set numpy random seed
np.random.seed(0)

scatter_size = 25
fig_size_width = 2
fig_size_height = 4

#set font size: x-axis, y-axis, title, legend
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.figsize'] = (fig_size_width, fig_size_height)




def read_input_file(file_loc):

    #read the input file using pandas
    agent_data = pd.read_csv(file_loc)

    # Define the EPSG codes for the source (3857) and target (4326) coordinate systems
    source_crs = pyproj.CRS.from_epsg(3857)
    target_crs = pyproj.CRS.from_epsg(4326)

    # Create a PyProj transformer to perform the coordinate conversion
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

    # Convert the coordinates from EPSG:3857 to EPSG:4326
    agent_data['lon_epsg3857'], agent_data['lat_epsg3857'] = transformer.transform(agent_data['longitude'].values, agent_data['latitude'].values)

    return agent_data

def calculate_MCP(agent_data):

    # Create a numpy array from latitude and longitude data
    points = np.array(list(zip(agent_data["lon_epsg3857"], agent_data['lat_epsg3857'])))

    # Calculate the convex hull
    hull = ConvexHull(points)

    # Extract the convex hull vertices
    hull_points = points[hull.vertices]

    import pyproj
    from shapely.geometry import shape
    from shapely.ops import transform

    #create a shapely geometry object from the convex hull vertices
    geom = {'type': 'Polygon', 'coordinates': [hull_points]}

    s = shape(geom)
    wgs84 = pyproj.CRS('EPSG:4326')
    utm = pyproj.CRS('EPSG:3857')
    project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    projected_area = transform(project, s).area

    # print("Area in square kilometers (MCP): ", projected_area*1e-6)

    #-----------------------------------------------------------------------
    # Calculate the area of the convex hull in square degrees
    convex_hull_area_deg2 = hull.volume

    # Latitude of Pathanamthitta, Kerala (approximately)
    latitude_pathanamthitta = min(agent_data['lat_epsg3857'])-0.025 + (max(agent_data['lat_epsg3857'])+0.025 - min(agent_data['lat_epsg3857'])-0.025)/2

    # Calculate the conversion factor
    conversion_factor = 111.32 / np.cos(np.radians(latitude_pathanamthitta))

    convex_hull_area_km2 = convex_hull_area_deg2 * conversion_factor**2

    # Print the area in square kilometers
    # print(f'MCP Area: {convex_hull_area_km2:.2f} square kilometers')
    #-----------------------------------------------------------------------

    return convex_hull_area_km2

def calculate_KDE(agent_data, bandwidth, contour_level, fig_background, ax_background):
    
    # Create a numpy array from latitude and longitude data
    points = np.array(list(zip(agent_data["lon_epsg3857"], agent_data['lat_epsg3857'])))

    # Calculate the convex hull
    hull = ConvexHull(points)

    # Extract the convex hull vertices
    hull_points = points[hull.vertices]
    
    X_train = np.vstack([agent_data["lat_epsg3857"], agent_data['lon_epsg3857']]).T
    X_train *= np.pi / 180.0  # Convert lat/long to radians

    kde = KernelDensity(
            bandwidth=bandwidth, metric="haversine", kernel="gaussian", algorithm="ball_tree"
        )
    kde.fit(X_train) 

    def parallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
        with multiprocessing.Pool(thread_count) as p:
            return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))

    # x coordinates of the grid cells
    xgrid = np.linspace(min(agent_data["lon_epsg3857"])-0.05, max(agent_data['lon_epsg3857'])+0.05, 1000)  #longitude

    # y coordinates of the grid cells
    ygrid = np.linspace(min(agent_data['lat_epsg3857'])-0.05, max(agent_data['lat_epsg3857'])+0.05, 1000)      #latitude

    X, Y = np.meshgrid(xgrid, ygrid)
    xy = np.vstack([Y.ravel(), X.ravel()]).T
    xy = np.radians(xy)
    Z = np.full(xy.shape[0], 0.0)

    # #Z[mask] = np.exp(kde.score_samples(xy))
    Z = np.exp(parallel_score_samples(kde, xy))

    # #convert to probability density
    Z = Z / Z.sum()
    Z = Z.reshape((1000, 1000))

    # contour_density = ax_background.contourf(X, Y, Z, cmap='YlGnBu', levels=10, alpha=0.5)

    # get contour lines from the contour plot to get the polygon coordinates
    contour_lines = ax_background.contour(X, Y, Z, 
                                          colors='black', 
                                          linewidths=1.5, 
                                          levels=10, 
                                          label = "KDE Contour")
    
    fmt = {}
    strs = [0, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    len(contour_lines.levels)
    for l, s in zip(contour_lines.levels, strs):
        fmt[l] = s

    ax_background.clabel(contour_lines, contour_lines.levels, 
                         inline=True, 
                         fontsize=16, 
                         fmt = fmt)

    # Find the contour level value
    sorted_density_values = np.sort(Z.ravel())      #sort the density values

    #get unique values
    sorted_density_values = np.unique(sorted_density_values)

    #choose every 500th value
    sorted_density_values = sorted_density_values[::500]

    # print("length of sorted_density_values: ", len(sorted_density_values))

    probability_list = []
    contour_values = []

    #for each density value, calculate the sum of all the values greater than or equal to it inthe Z array
    for id, density_value in enumerate(sorted_density_values):
        mask = Z >= density_value
        sum = Z[mask].sum()
        # print("id", id, "density_value: ", density_value, "probability: ", 1 - sum)
        probability_list.append(1 - sum)
        contour_values.append(density_value)

    #choose the probability value closest to the contour level value given by the user
    contour_value = np.interp(contour_level, probability_list, contour_values)      #interpolate the contour value
    # print("Contour value for the given quantile: ", contour_value)

    # Mask density values below the contour level
    masked_density = np.ma.masked_less(Z, contour_value)        #mask the density values below the contour level

    # PLot the MCP
    for simplex in hull.simplices:
        mcp = ax_background.plot(points[simplex, 0], 
                                 points[simplex, 1], 
                                 'magenta', 
                                 linewidth=2.5,
                                 label = "MCP")

    import pyproj
    from shapely.geometry import shape
    from shapely.ops import transform

    for i, path in enumerate(contour_lines.collections):
        level = contour_lines.levels[i]
        # print("level: ", level)

        total_area_in_level = 0

        for polygon in path.get_paths():
            vertices = polygon.vertices
            geom = {'type': 'Polygon', 'coordinates': [vertices]}
            s = shape(geom)

            #convert the coordinates to EPSG:4326
            wgs84 = pyproj.CRS('EPSG:4326')
            utm = pyproj.CRS('EPSG:3857')
            project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
            projected_area = transform(project, s).area
            total_area_in_level += projected_area

        if i == 2:
            # print("KDE area: ", total_area_in_level*1e-6)
            KDE_area = total_area_in_level*1e-6


    # calculate_MCP(agent_data)

    # ax_background.plot(lon, lat, color="orangered", linewidth=.30, zorder=2)
    start = ax_background.scatter(0, 0, 75, marker='^', color='red', zorder=2, label = "Start", edgecolors='black', linewidths=1)
    end = ax_background.scatter(0, 0, 75, marker='v', color='red', zorder=2, label = "End", edgecolors='black', linewidths=1)

    plt.legend(handles=[mcp[0], start, end], loc="upper right")
    
    return KDE_area




#------------------------------------------------------------------------------------------------
class PlotDayNightTrajSequence:
    """
    Summary: This class takes an experiment ID and plots a day-night trajectory sequence of the elephant agent

    """
    #------------------------------------------------------------------------------------------------
    def __init__(self, folder):
        """
        Summary: Initialises the class

        Args:

        scenario_food_val (int): Scenario food value
        thermoregulation_threshold (int): Temperature threshold
        elephant_category (str): Elephant category
        aggression (float): Aggression value
        model_ID (str): Behavioural model ID
        month (str): Month
        simulation_ID (str): Simulation ID

        """

        self.input_folder = os.path.join("sensitivity_analysis/outputs/forest-food-percent")
        self.folder = folder

        self.show_all_simulations()
    #------------------------------------------------------------------------------------------------
    def show_all_simulations(self):
        """
        Summary: Shows all simulations in the folder
        """

        # self.MCP = []
        # self.KDE = []

        self.diel_distance = []
        self.net_distance = []

        # print(os.listdir(os.path.join(self.input_folder, self.folder)))

        folders = os.listdir(os.path.join(self.input_folder, self.folder))

        #remove the folder "food_and_water_matrix" from the list
        folders = [folder for folder in folders if "food_and_water_matrix" not in folder]

        for folder in folders:
            # MCP_AREA, KDE_AREA = self.calculate_space_use(folder)
            # self.MCP.append(MCP_AREA)
            # self.KDE.append(KDE_AREA)

            diel_distance, net_distance = self.calculate_diel_and_net_displacement(folder)
            self.diel_distance.append(diel_distance)
            self.net_distance.append(net_distance)
    #------------------------------------------------------------------------------------------------
    def calculate_space_use(self, folder):
        """
        """

        self.pathtofile = os.path.join(self.input_folder, self.folder, folder, "output_files", "agent_data.csv")
    
        agent_data  = read_input_file(self.pathtofile)

        #sample evry 12th step
        agent_data = agent_data.iloc[::12, :]

        agent = agent_data["AgentID"].unique()[0]

        if "herd" in agent or "bull" in agent:

            agent_data = agent_data[agent_data["AgentID"] == agent]
            lon = agent_data["longitude"]
            lat = agent_data["latitude"]

            outProj, inProj =  Proj(init='epsg:4326'),Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
            x_new, y_new = transform(inProj, outProj, lon, lat)

            MCP_AREA = calculate_MCP(agent_data)

            fig_background, ax_background = plt.subplots(figsize = (fig_size_width, fig_size_height))
            KDE_AREA = calculate_KDE(agent_data, 0.000075, 0.95, fig_background, ax_background)

            return MCP_AREA, KDE_AREA
    #------------------------------------------------------------------------------------------------
    def calculate_diel_and_net_displacement(self, folder):

        self.pathtofile = os.path.join(self.input_folder, self.folder, folder, "output_files", "agent_data.csv")
    
        agent_data  = read_input_file(self.pathtofile)

        agent = agent_data["AgentID"].unique()[0]

        if "herd" in agent or "bull" in agent:

            agent_data = agent_data[agent_data["AgentID"] == agent]
            lon = agent_data["longitude"]
            lat = agent_data["latitude"]

            outProj, inProj =  Proj(init='epsg:4326'), Proj(init='epsg:3857')   #projection to the CRS on which mesa runs
            lon, lat = transform(inProj, outProj, lon, lat)

            agent_data["longitude"] = lon
            agent_data["latitude"] = lat

            import geopy.distance # type: ignore

            dist = []

            for i in range(0, len(agent_data) - 1):
                coords_1 = [ lat[i], lon[i]]
                coords_2 = [ lat[i+1], lon[i+1]]
                dist.append(geopy.distance.geodesic(coords_1, coords_2).km)
            dist.append(np.nan)

            agent_data["step"] = dist

            for i in range(0, len(agent_data)-288, 288):

                day_data=agent_data[i*288:i*288+288]

                if len(day_data) == 288:

                    diel_distance = sum(day_data["step"])

                    coords_1 = [ day_data.head(1)["latitude"].values[0], day_data.head(1)["longitude"].values[0]]
                    coords_2 = [ day_data.tail(1)["latitude"].values[0], day_data.tail(1)["longitude"].values[0]]

                    # print("coords_1: ", coords_1)
                    # print("coords_2: ", coords_2)
                    
                    net_distance = geopy.distance.geodesic(coords_1, coords_2).km

            print("Net distance: ", net_distance)
            print("Diel distance: ", diel_distance)

        return diel_distance, net_distance


#----------------------------------------------------------------------------------------------------
def main():

    prob_water = 0.0
    food_val_cropland = 100
    threshold = 32
    year = 2010

    for food_val_forest in [5, 10, 15, 20, 25]:
        for aggression in [0.2, 0.8]:
            for month_idx in [3, 8]:

                diel_displacemet_list = []
                net_displacemet_list = []

                for prob_food_forest in [0.08, 0.09, 0.1, 0.11]:

                    month = calendar.month_abbr[month_idx]
                    num_days = calendar.monthrange(year, month_idx)[1]

                    expt_name = "forest-food-percent:" + str(prob_food_forest)
                    output = "prob_water__" + str(prob_water) + "__output_files"
                    food_val = "food_value_forest__" + str(food_val_forest) + "__food_value_cropland__" + str(food_val_cropland)
                    temp_threshold = "THRESHOLD_" + str(threshold)
                    elephant_category = "solitary_bulls"
                    expt_id = "aggression:" + str(aggression)
                    
                    folder = os.path.join(os.getcwd(), "sensitivity_analysis/outputs/forest-food-percent/", expt_name, output, food_val, temp_threshold, elephant_category, expt_id, "model_01", month)
                    
                    my_plot = PlotDayNightTrajSequence(folder)

                    diel_displacemet_list.append(my_plot.diel_distance)
                    net_displacemet_list.append(my_plot.net_distance)

                #plot the box plots
                fig, ax = plt.subplots()
                ax.boxplot(diel_displacemet_list, positions = [1, 2, 3, 4], widths = 0.5, patch_artist = True)
                ax.set_xlabel("Probability of food in forest")
                ax.set_ylabel("Area (sq. km)")
                ax.set_xticks([1, 2, 3, 4])
                ax.set_xticklabels(["0.08", "0.09", "0.1", "0.11"])
                plt.savefig("sensitivity_analysis/plots/forest-food-percent/codes/diel_and_net_distance_plots/diel_food_val:" + str(food_val_forest) + "_agg:" + str(aggression) + "_mon:" + str(month_idx) + ".png", dpi=300, bbox_inches='tight')

                fig, ax = plt.subplots()
                ax.boxplot(net_displacemet_list, positions = [1, 2, 3, 4], widths = 0.5, patch_artist = True)
                ax.set_xlabel("Probability of food in forest")
                ax.set_ylabel("Area (sq. km)")
                ax.set_xticks([1, 2, 3, 4])
                ax.set_xticklabels(["0.08", "0.09", "0.1", "0.11"])
                plt.savefig("sensitivity_analysis/plots/forest-food-percent/codes/diel_and_net_distance_plots/net_food_val:" + str(food_val_forest) + "_agg:" + str(aggression) + "_mon:" + str(month_idx) + ".png", dpi=300, bbox_inches='tight')

                plt.close('all')
#----------------------------------------------------------------------------------------------------  
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------------------------------


