import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon as ShapelyPolygon


regions_coordinates = [
    {  # Region: PECK
        "name": "PECK",
        "coords": [(-85.03859, 43.58688), (-82.22609, 43.97555), (-82.1382, 43.59484), (-82.44582, 42.99513), (-83.37966, 42.6971), (-85.07155, 42.76166)]
    },
    {  # Region: DETROIT
        "name": "DETROIT",
        "coords": [(-85.07155, 42.76166), (-83.37966, 42.6971), (-82.7864, 41.87624), (-84.66506, 41.85169), (-84.98366, 42.03147)]
    },
    {  # Region: BLUFFTON
        "name": "BLUFFTON",
        "coords": [(-84.66506, 41.85169), (-82.7864, 41.87624), (-82.66555, 41.17713), (-83.43459, 40.7374), (-84.73098, 40.94519)]
    },
    {  # Region: HUDSON
        "name": "HUDSON",
        "coords": [(-82.44582, 42.99513), (-83.37966, 42.6971), (-82.7864, 41.87624), (-81.28127, 41.83532), (-80.90773, 42.51111), (-80.49025, 42.83421)]
    },
    {  # Region: RAVENNA
        "name": "RAVENNA",
        "coords": [(-82.7864, 41.87624), (-81.28127, 41.83532), (-80.55617, 41.69602), (-80.7759, 41.10267), (-80.27053, 40.66244), (-80.76491, 40.69577), (-81.63283, 40.58739), (-83.43459, 40.7374), (-82.66555, 41.17713)]
    },
    {  # Region: NIAGARA
        "name": "NIAGARA",
        "coords": [(-80.90773, 42.51111), (-80.49025, 42.83421), (-79.35866, 43.05938), (-78.79836, 42.9469), (-77.04055, 42.49491), (-77.3152, 42.04779), (-79.14992, 42.4544)]
    },
    {  # Region: JAMESTOWN
        "name": "JAMESTOWN",
        "coords": [(-80.90773, 42.51111), (-81.28127, 41.83532), (-80.55617, 41.69602), (-77.67775, 41.46591), (-77.3152, 42.04779), (-79.14992, 42.4544)]
    },
    {  # Region: CLARKSBURG
        "name": "CLARKSBURG",
        "coords": [(-80.55617, 41.69602), (-77.67775, 41.46591), (-77.93044, 41.02812), (-77.76564, 40.64577), (-80.27053, 40.66244), (-80.7759, 41.10267)]
    },
    {  # Region: FRANKLIN
        "name": "FRANKLIN",
        "coords": [(-81.63283, 40.58739), (-80.76491, 40.69577), (-78.40285, 39.67181), (-80.2925, 39.11997), (-81.38015, 39.82386)]
    },
    {  # Region: ALLEGHENY
        "name": "ALLEGHENY",
        "coords": [(-80.76491, 40.69577), (-80.27053, 40.66244), (-77.76564, 40.64577), (-77.58986, 40.05971), (-78.40285, 39.67181)]
    },
    {  # Region: GENESEO
        "name": "GENESEO",
        "coords": [(-79.35866, 43.05938), (-78.79836, 42.9469), (-77.04055, 42.49491), (-76.15065, 42.65672), (-76.87575, 43.84099), (-78.77639, 43.7617)]
    }
]

airport_coordinates = {
    "KFNT": (42.9654, -83.7436),  # Flint, MI
    "KDTW": (42.2125, -83.3534),  # Detroit, MI
    "KTOL": (41.5868, -83.8078),  # Toledo, OH
    "KCLE": (41.4118, -81.8498),  # Cleveland, OH
    "KPIT": (40.4915, -80.2329),  # Pittsburgh, PA
    "KBUF": (42.9405, -78.7322),  # Buffalo, NY
    "KROC": (43.1189, -77.6724),  # Rochester, NY
    "Columbus": (39.9612, -82.9988)  # Columbus, OH
}


def plot_region_track(df, plane_id, output_path):

    df = df[df['acId'] == plane_id]
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    ax.set_extent([-90, -75, 38, 45], ccrs.Geodetic())

    ax.add_feature(cfeature.BORDERS, edgecolor='black')
    ax.add_feature(cfeature.STATES, edgecolor='black')

    for region in regions_coordinates:
        coords = region["coords"]
        name = region["name"]
        polygon = ShapelyPolygon(coords)
        x, y = polygon.exterior.xy
        ax.plot(x, y, transform=ccrs.Geodetic(), color='red', linewidth=2)
        centroid = polygon.centroid
        ax.text(centroid.x, centroid.y, name, fontsize=12, ha='center', va='center', color='red', transform=ccrs.Geodetic())


    ac_df = df.sort_values(by='recTime')  
    longitude = ac_df['coord2'].astype(float)  
    latitude = ac_df['coord1'].astype(float)   
    ax.plot(longitude, latitude, label=plane_id, linewidth=0.5)  
    ax.scatter(longitude, latitude, s=10, edgecolor='k', zorder=5) 


    for airport, (lat, lon) in airport_coordinates.items():
        ax.plot(lon, lat, marker='o', color='blue', markersize=10, transform=ccrs.Geodetic())
        ax.text(lon, lat, airport, fontsize=9, ha='right', va='bottom', color='blue', transform=ccrs.Geodetic())
    
    
    ax.legend(loc="upper left", title="AC ID", bbox_to_anchor=(1.05, 1))

    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', zorder=3)
    
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')


    plt.title(f'Flight Tracks for Plane {plane_id}')

    
    plt.savefig(output_path)
    plt.close()
    
if __name__ == "__main__":
    filtered_rows_path = 'output/filtered_rows.csv'
    filtered_rows = pd.read_csv(filtered_rows_path)

    