# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:27:01 2024

@author: llama
"""


import pandas as pd
import numpy as np
#import random
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

#Counterclockwise rotation that fixes the y-axis
def rot_fix_y(angle_rad,x,y,z):
    return (x*math.cos(angle_rad)-z*math.sin(angle_rad),
            y,
            x*math.sin(angle_rad)+z*math.cos(angle_rad))

#Counterclockwise rotation that fixes the z-axis
def rot_fix_z(angle_rad,x,y,z):
    return (x*math.cos(angle_rad)-y*math.sin(angle_rad),
            x*math.sin(angle_rad)+y*math.cos(angle_rad),
            z)

#Transform radians to degrees
def radian2degree(radians):
    return radians*180/math.pi

#Transform degrees to radians
def degree2radian(degrees):
    return degrees*math.pi/180



#Transform spherical coordinates to cartesian coordinates
def spherical2cartesian(r, latitude, longitude):
    return (r*math.cos(latitude)*math.cos(longitude), r*math.cos(latitude)*math.sin(longitude), r*math.sin(latitude))

#Transform cartesian coordinates to spherical coordinates
def cartesian2spherical(x,y,z):
    r=math.sqrt(x**2+y**2+z**2)
    if r==0:
        return (0,0,0)
    latitude=math.asin(z/r)
    if x==0 and y==0:
        return (r,latitude,0)
    longitude=0
    if y>0:
        longitude=math.acos(x/math.sqrt(x**2+y**2))
    else:
        longitude=-math.acos(x/math.sqrt(x**2+y**2))
    return (r, latitude, longitude)

#paralles
def draw_parallels(mapa):
    for latitude in range(-90, 91, 10):
        parallel=[]
        for longitude in range(-180, 181, 10):
            parallel.append([latitude,longitude])
        folium.PolyLine(
        locations=parallel,
        color="black",
        weight=1,
        tooltip=str(latitude)+"º",
        ).add_to(mapa)
    
#meridians
def draw_meridians(mapa):
    for longitude in range(-180, 181, 10):
        meridian=[]
        for latitude in range(-90, 91, 10):
            meridian.append([latitude,longitude])
        folium.PolyLine(
        locations=meridian,
        color="black",
        weight=1,
        tooltip=str(longitude)+"º",
        ).add_to(mapa)    

# Draws lines that are at the same distance from a given point (like parallels are at the same distance from the North Pole).
def draw_distances_to_origin(latitude_rad, longitude_rad, mapa):
    for i in range(19):
        x=math.cos(degree2radian(i*10))
        coor=[]
        for angle in np.arange(0,2*math.pi,0.01):
            (x,y,z)=(x,(1-x**2)*math.cos(angle),(1-x**2)*math.sin(angle))
            (x1,y1,z1)=rot_fix_y(latitude_rad,x,y,z)
            (x2,y2,z2)=rot_fix_z(longitude_rad,x1,y1,z1)
            (r, latitude2_rad, longitude2_rad)=cartesian2spherical(x2,y2,z2)
            coor.append([radian2degree(latitude2_rad),radian2degree(longitude2_rad)])
            
            folium.CircleMarker(
            location=[radian2degree(latitude2_rad),radian2degree(longitude2_rad)],
            color="red",
            tooltip=str(i),
            radius=2,
            weight=5,
            ).add_to(mapa)

  
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

      
draw_parallels(mapa)
draw_meridians(mapa)
draw_distances_to_origin(degree2radian(coordinates_origin[0]) ,degree2radian(coordinates_origin[1]), mapa)

# locations=[]
# for same_distance in distances:
    
#         angles={}
#         for airport in distances[same_distance]:
#             coordinates_destination=[float(data[airport]['latitude']), float(data[airport]['longitude'])]
#             coordinates_destination2=[coordinates_destination[0]-coordinates_origin[0], coordinates_destination[1]-coordinates_origin[1]]
#             angles[airport] = math.atan2(coordinates_destination2[0],coordinates_destination2[1])
            
#         sorted_airports = dict(sorted(angles.items(), key=lambda item: item[1]))
        
        
#         locations.append([])
#         i=1
#         for airport in sorted_airports:
#             coordinates_destination=[float(data[airport]['latitude']), float(data[airport]['longitude'])]
#             folium.CircleMarker(location=coordinates_destination,
#                                   tooltip=str(i),
#                                   radius=1,
#                                   weight=5).add_to(mapa)
            
#             i=i+1
#             locations[len(locations)-1].append(coordinates_destination)
#         # folium.Polygon(
#         #     locations=locations[len(locations)-1],
#         #     color="blue",
#         #     weight=1,
#         #     fill_color="red",
#         #     fill_opacity=0.01,
#         #     fill=True,
#         #     tooltip=str(same_distance*15)+"min"
#         # ).add_to(mapa)
   
# folium.CircleMarker(location=coordinates_destination,
#                       radius=1,
#                       weight=5).add_to(mapa)

# folium.CircleMarker(location=coordinates_destination2,
#                       radius=1,
#                       color='red',
#                       tooltip=str(coordinates_destination2) + str(math.atan2(coordinates_destination2[0],coordinates_destination2[1])),
#                       weight=5).add_to(mapa)
    
# folium.CircleMarker(location=[0,0],
#                           radius=1,
#                           color='red',
#                           weight=5).add_to(mapa)    




mapa.save("map.html")