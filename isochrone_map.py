# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:27:01 2024

@author: llama
"""


import pandas as pd
import random
import json
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
import math

norm = colors.Normalize(vmin=0, vmax=600)
f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('magma'))

def f2hex(f2rgb, f):
    rgb = f2rgb.to_rgba(f)[:3]
    return '#%02x%02x%02x' % tuple([int(255*fc) for fc in rgb])




# Open and read the JSON file
with open('airline_routes.json', 'r') as file:
    data = json.load(file)

airports = pd.read_excel('airports_code.xlsx')
continents=pd.read_excel('continents.xlsx')
airports = airports.set_index('Country Name').join(continents.set_index('Country'), how='left').reset_index().rename(columns={'index': 'country'})


origin_airport='FRA'

coordinates_origin=[float(data[origin_airport]['latitude']),float(data[origin_airport]['longitude'])]
    
mapa = folium.Map(location=coordinates_origin, zoom_start=8)

folium.CircleMarker(location=coordinates_origin,
                        radius=2,
                        color="#7aff33",
                        weight=5).add_to(mapa)

distances={}

for destination in data[origin_airport]['routes']:
    coordinates_destination=[float(data[destination['iata']]['latitude']), float(data[destination['iata']]['longitude'])]
    
    # folium.PolyLine(
    # locations=[coordinates_origin, coordinates_destination],
    # color="#FF0000",
    # weight=5,
    # tooltip="From "+data[origin_airport]['city_name']+" to "+data[destination['iata']]['city_name']
    # ).add_to(mapa)
   
    # folium.CircleMarker(location=coordinates_destination,
    #                     color=f2hex(f2rgb, destination['min']),
    #                     radius=1,
    #                     weight=5).add_to(mapa)
    d=destination['min']//15
    if d in distances:
        distances[d].append(destination['iata'])
    else:
        distances[d]=[destination['iata']]

locations=[]
for same_distance in distances:
    if same_distance<10:
        angles={}
        for airport in distances[same_distance]:
            coordinates_destination=[float(data[airport]['latitude']), float(data[airport]['longitude'])]
            coordinates_destination2=[coordinates_destination[0]-coordinates_origin[0], coordinates_destination[1]-coordinates_origin[1]]
            angles[airport] = math.atan2(coordinates_destination2[0],coordinates_destination2[1])
            
        sorted_airports = dict(sorted(angles.items(), key=lambda item: item[1]))
        
        
        locations.append([])
        i=1
        for airport in sorted_airports:
            coordinates_destination=[float(data[airport]['latitude']), float(data[airport]['longitude'])]
            folium.CircleMarker(location=coordinates_destination,
                                  tooltip=str(i),
                                  radius=1,
                                  weight=5).add_to(mapa)
            
            i=i+1
            locations[len(locations)-1].append(coordinates_destination)
        folium.Polygon(
            locations=locations[len(locations)-1],
            color="blue",
            weight=1,
            fill_color="red",
            fill_opacity=0.01,
            fill=True,
            tooltip=str(same_distance*15)+"min"
        ).add_to(mapa)
   
# folium.CircleMarker(location=coordinates_destination,
#                      radius=1,
#                      weight=5).add_to(mapa)

# folium.CircleMarker(location=coordinates_destination2,
#                      radius=1,
#                      color='red',
#                      tooltip=str(coordinates_destination2) + str(math.atan2(coordinates_destination2[0],coordinates_destination2[1])),
#                      weight=5).add_to(mapa)
    
# folium.CircleMarker(location=[0,0],
#                          radius=1,
#                          color='red',
#                          weight=5).add_to(mapa)    




mapa.save("map.html")