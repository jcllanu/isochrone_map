# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:27:01 2024

@author: llama
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
import math


class Polygon:
    
    def __init__(self,vertexes):
        self.vertexes = vertexes
        self.sides = len(vertexes)
        
    def edges(self):
        edges=[]
        for i in range(self.sides):
            edges.append(Segment(self.vertexes[i],self.vertexes[(i+1)%self.sides]))
        return edges
    def is_point_inside(self, point):
        count = 0
        x, y = point[0], point[1]
        
        for edge in polygon.edges():
            x1, y1 = edge.origin[0], edge.origin[1]
            x2, y2 = edge.end[0], edge.end[1]
            
            # Check if the point is exactly on an edge
            if min(y1, y2) <= y <= max(y1, y2) and min(x1, x2) <= x <= max(x1, x2):
                if (x2 - x1) * (y - y1) == (x - x1) * (y2 - y1):  # Cross product == 0
                    return True  # On the edge
    
            # Check if horizontal ray to the right intersects the edge
            if y1 > y2:
                x1, x2, y1, y2 = x2, x1, y2, y1
            if y == y1 or y == y2:  # Avoid edge case where ray hits vertex
                y += 0.00001
            if y1 < y < y2:  # horizontal line must intersect edge, but we have to check if it is on the right
                x_intersection = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x_intersection > x:
                    count += 1
        return count % 2 == 1
class Segment:
    def __init__(self,origin, end):
        self.origin=origin
        self.end=end
    def intersectsSegment(self,other):
        (x1,y1)=self.origin
        (u1,v1)=self.end
        # equation of the first segment: (x1 + t*(u1-x1), y1 + t*(v1-y1)), t in [0,1]
        (x2,y2)=other.origin
        (u2,v2)=other.end
        # equation of the second segment: (x2 + s*(u2-x2), y2 + s*(v2-y2)), s in [0,1]
        
        # x1 + t*(u1-x1) = x2 + s*(u2-x2) <-> (u1-x1)*t + (x2-u2)*s = x2-x1
        # y1 + t*(v1-y1) = y2 + s*(v2-y2) <-> (v1-y1)*t + (y2-v2)*s = y2-y1
        a = np.array([[u1-x1, x2-u2], [v1-y1, y2-v2]])
        b = np.array([x2-x1, y2-y1])
        try:
            x=np.linalg.solve(a,b)
        except:
            return (False, 0)
        else:
            if 0 <= x[0] and 0 <= x[1] and x[0] <= 1 and x[1] < 1:
                return (True, x[0])
            else:
                return (False, 0)
            
    def intersectsPolygon(self,polygon):
        minimum=2
        positive_minimum=2
        count=0
        segment_int=[]
        segment_int_positive=[]
        for segment in polygon.edges():
            intersection = self.intersectsSegment(segment)
            if intersection[0]:
                count=count+1
                if intersection[1] < minimum:
                    minimum = intersection[1]
                    segment_int=segment
                if 0 < intersection[1] < positive_minimum:
                    positive_minimum = intersection[1]
                    segment_int_positive=segment
        if count==0: # no intersections
            return (False,[], minimum, count)
        elif minimum==0:
            if count==1:
                return (False,[], minimum, count)
            elif count % 2 == 0:
                s=Segment(self.origin, segment_int.end)
                (boolean, seg, m, c) = s.intersectsPolygon(polygon)
                if c>1:
                    return (True, seg, minimum, count)
                else:
                    return (True,segment_int, minimum, count)
            else:
                s=Segment(self.origin, segment_int_positive.end)
                (boolean, seg, m, c) = s.intersectsPolygon(polygon)
                if c>1:
                    return (True,seg, minimum, count)
                else:
                    return (True,segment_int_positive, minimum, count)
        else:
            s=Segment(self.origin, segment_int.end)
            (boolean, seg, m, c) = s.intersectsPolygon(polygon)
            if c>1:
                return (True,seg, minimum, count)
            else:
                return (True,segment_int, minimum, count)
        
norm = colors.Normalize(vmin=0, vmax=600)
f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('magma'))

def unique_values(l):
    unique_values = []
    for value in l:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values
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

def radial_projection(x,y,z):
    if x>0:
        return (y/x, z/x)
    
def inverse_radial_projection(u,v):
    r=math.sqrt(1+u**2+v**2)
    return (1/r, u/r, v/r)
    
def parallel_projection(x,y,z):
    return (y, z)

#Transform polar coordinates to cartesian coordinates
def polar2cartesian(r, alpha):
    return (r*math.cos(alpha), r*math.sin(alpha))

#Transform cartesian coordinates to polar coordinates
def cartesian2polar(x,y):
    r=math.sqrt(x**2+y**2)
    alpha = np.arctan2(y, x)
    return(r, alpha)

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
    

distances={}
distance_from_airport={}
for destination in data[origin_airport]['routes']:
    coordinates_destination=[float(data[destination['iata']]['latitude']), float(data[destination['iata']]['longitude'])]
    
    # draw line from origin to destination
    # folium.PolyLine(
    # locations=[coordinates_origin, coordinates_destination],
    # color="#FF0000",
    # weight=5,
    # tooltip="From "+data[origin_airport]['city_name']+" to "+data[destination['iata']]['city_name']
    # ).add_to(mapa)
    
    # draw destination point
    # folium.CircleMarker(location=coordinates_destination,
    #                     color=f2hex(f2rgb, destination['min']),
    #                     radius=1,
    #                     weight=5).add_to(mapa)
    d=destination['min']//30
    distance_from_airport[destination['iata']]=d
    if d in distances:
        distances[d].append(destination['iata'])
    else:
        distances[d]=[destination['iata']]

      


perimeters=[]
XY_plane={}
XY_plane[origin_airport]=(0,0)
airport_from_coors={}
polygon=[]

sorted_distances = dict(sorted(distances.items(), key=lambda item: item[0]))
for same_distance in sorted_distances:
    angles={}
    for airport in distances[same_distance]:
        coordinates_destination=[float(data[airport]['latitude']),float(data[airport]['longitude'])]
        
        # folium.CircleMarker(
        # location=coordinates_destination,
        # color="red",
        # tooltip=airport,
        # radius=2,
        # weight=5,
        # ).add_to(mapa)
        
        # apply rotation that moves the origin to (x,y,z)=(1,0,0)
        (x,y,z)=spherical2cartesian(1, degree2radian(coordinates_destination[0]), degree2radian(coordinates_destination[1]))
        (x1,y1,z1)=rot_fix_z(-degree2radian(coordinates_origin[1]),x,y,z)
        (x2,y2,z2)=rot_fix_y(-degree2radian(coordinates_origin[0]),x1,y1,z1)
        
        if x2<0:
            continue
        (u,v)=radial_projection(x2,y2,z2)
        (rho,alpha)=cartesian2polar(u,v)
        angles[airport]=alpha
        XY_plane[airport]=(u,v)
        airport_from_coors[(u,v)]=airport
        (r, latitude2_rad, longitude2_rad)=cartesian2spherical(x2,y2,z2)
        
        # folium.CircleMarker(
        # location=[radian2degree(latitude2_rad),radian2degree(longitude2_rad)],
        # color="red",
        # tooltip=airport,
        # radius=2,
        # weight=5,
        # ).add_to(mapa)
        
    if len(angles)==0:
        continue
    airports_sorted = dict(sorted(angles.items(), key=lambda item: item[1]))
    sorted_airports=list(airports_sorted.keys())
    sorted_airports.append(sorted_airports[0])
    x_coords=[XY_plane[airport][0] for airport in sorted_airports]
    y_coords=[XY_plane[airport][1] for airport in sorted_airports] 
    # plt.plot(x_coords, y_coords, 'o')  
    # for i, label in enumerate(sorted_airports):
    #     plt.text(x_coords[i], y_coords[i], label, fontsize=12, ha='right')
    # plt.show()                           
    if len(perimeters)==0:
        polygon=Polygon([XY_plane[airport] for airport in sorted_airports])
        if polygon.is_point_inside([0,0]):
            perimeters.append(sorted_airports)
        else:
            # Include origin in the set of points and compute centroid of extended list
            sorted_airports.append(origin_airport)
            distance_from_airport[origin_airport]=distance_from_airport[sorted_airports[0]]
            x_coords.append(0)
            y_coords.append(0)
            centroid=[sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)]
            
            #Change of coordinates so that centroid is the new origin
            angles={}
            for airport in sorted_airports:
                (rho,alpha)=cartesian2polar(XY_plane[airport][0]-centroid[0], XY_plane[airport][1]-centroid[1])
                angles[airport]=alpha

            # Sort by angle
            airports_sorted = dict(sorted(angles.items(), key=lambda item: item[1]))
            sorted_airports=list(airports_sorted.keys())
            sorted_airports.append(sorted_airports[0])
            x_coords=[XY_plane[airport][0] for airport in sorted_airports]
            y_coords=[XY_plane[airport][1] for airport in sorted_airports] 
            # plt.plot(x_coords, y_coords, 'o')  
            # for i, label in enumerate(sorted_airports):
            #     plt.text(x_coords[i], y_coords[i], label, fontsize=12, ha='right')
            # plt.show()
            
            polygon=Polygon([XY_plane[airport] for airport in sorted_airports])
            perimeters.append(sorted_airports)
    else:
        new_perimeter=[]
        for airport in sorted_airports:
            if polygon.is_point_inside(XY_plane[airport]):
                 continue
            if len(new_perimeter)>0:
                segment=Segment(XY_plane[new_perimeter[-1]], XY_plane[airport])
                intersects=segment.intersectsPolygon(polygon)
                old_perimeter = list(perimeters[-1])           
                old_perimeter = unique_values(old_perimeter)
                maximum = 0
                if intersects[0]:
                    old_perimeter=old_perimeter[old_perimeter.index(airport_from_coors[intersects[1].end]):]+old_perimeter[:old_perimeter.index(airport_from_coors[intersects[1].end])]
                while intersects[0]:
                    print("Trying: "+new_perimeter[-1]+" to "+airport)
                    print("Intersects: "+airport_from_coors[intersects[1].origin]+" to "+airport_from_coors[intersects[1].end])
                    print()
                    
                    index = old_perimeter.index(airport_from_coors[intersects[1].end])
                    if index < maximum:
                        new_perimeter.append(old_perimeter[maximum])
                        maximum = maximum + 1
                    else:
                        new_perimeter.append(airport_from_coors[intersects[1].end])
                        maximum = index + 1
                    
                    segment=Segment(XY_plane[new_perimeter[-1]], XY_plane[airport])
                    intersects=segment.intersectsPolygon(polygon)
                new_perimeter.append(airport)
            else:
                new_perimeter.append(airport)
        polygon=Polygon([XY_plane[airport] for airport in new_perimeter])
        if polygon.is_point_inside([0,0]):
            perimeters.append(new_perimeter)
        else:
            # Include origin in the set of points and compute centroid of extended list
            sorted_airports = unique_values(new_perimeter)
            sorted_airports.append(origin_airport)
            x_coords.append(0)
            y_coords.append(0)
            centroid=[sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)]
            
            #Change of coordinates so that centroid is the new origin
            angles={}
            for airport in sorted_airports:
                (rho,alpha)=cartesian2polar(XY_plane[airport][0]-centroid[0], XY_plane[airport][1]-centroid[1])
                angles[airport]=alpha

            # Sort by angle
            airports_sorted = dict(sorted(angles.items(), key=lambda item: item[1]))
            sorted_airports=list(airports_sorted.keys())
            index=sorted_airports.index(origin_airport)
            airportA = sorted_airports[(index+1)%len(sorted_airports)] 
            airportB = sorted_airports[(index-1)%len(sorted_airports)]
            
            old_perimeter = list(perimeters[-1])
            polygon=Polygon([XY_plane[airport] for airport in old_perimeter])
            segmentA = Segment(XY_plane[airportA],[0,0])
            intersects = segmentA.intersectsPolygon(polygon)
            perimeter_airport_A = airport_from_coors[intersects[1].origin]
            
            segmentB = Segment(XY_plane[airportB],[0,0])
            intersects = segmentB.intersectsPolygon(polygon)
            perimeter_airport_B = airport_from_coors[intersects[1].end]
            
            
            old_perimeter = unique_values(old_perimeter)
            new_perimeter = unique_values(new_perimeter)
            
            index_perimeter_airport_A = old_perimeter.index(perimeter_airport_A)
            index_airport_A = new_perimeter.index(airportA)
            
            old_perimeter = old_perimeter[index_perimeter_airport_A:] + old_perimeter[:index_perimeter_airport_A]
            new_perimeter = new_perimeter[index_airport_A:] + new_perimeter[:index_airport_A]
            
            new_perimeter = [perimeter_airport_A] + new_perimeter[new_perimeter.index(airportA):new_perimeter.index(airportB)+1] + old_perimeter[old_perimeter.index(perimeter_airport_B):]+[perimeter_airport_A]
            perimeters.append(new_perimeter)
            polygon=Polygon([XY_plane[airport] for airport in new_perimeter])
    for perimeter in perimeters[-2:]:
        x_coords=[XY_plane[airport][0] for airport in perimeter]
        y_coords=[XY_plane[airport][1] for airport in perimeter] 
        plt.plot(x_coords, y_coords)  
        for i, label in enumerate(perimeter):
            plt.text(x_coords[i], y_coords[i], label, fontsize=12, ha='right')
    plt.show() 
            
def check_borders(locations):
    locations_corrected=[]
    for i in range(len(locations)):       
        origin=locations[i]
        end=locations[(i+1)%len(locations)]
        locations_corrected.append(origin)
        
        if origin[1]>175 and end[1]<-175:
            locations_corrected.append([89.99, 179.99])
            locations_corrected.append([89.99, -179.99])
        if origin[1]<-175 and end[1]>175:
            locations_corrected.append([89.99, -179.99])
            locations_corrected.append([89.99, 179.99])
    return locations_corrected
            
        
def print_shape(inner_perimeter, outter_perimeter, mapa, color, latitude_rad, longitude_rad):
    perimeter = outter_perimeter + [outter_perimeter[0]] + inner_perimeter[::-1]
    tmax=0
    for airport in outter_perimeter:
        coordinates_destination=[float(data[airport]['latitude']), float(data[airport]['longitude'])]
        folium.CircleMarker(location=coordinates_destination,
                              tooltip=airport,
                              radius=1,
                              weight=5).add_to(mapa)
        tmax=max(distance_from_airport[airport]*30,tmax)
    locations=[]   
    for i in range(len(perimeter)):
        airport_origin=perimeter[i]
        airport_end=perimeter[(i+1)%len(perimeter)]
        u1, v1, u2, v2 = XY_plane[airport_origin][0], XY_plane[airport_origin][1], XY_plane[airport_end][0], XY_plane[airport_end][1]
        N=50
        for t in range(N+1):
            u_t = u1 + t * (u2-u1)/N
            v_t = v1 + t * (v2-v1)/N
            (x,y,z) = inverse_radial_projection(u_t, v_t)
            (x1,y1,z1)=rot_fix_y(latitude_rad,x,y,z)
            (x2,y2,z2)=rot_fix_z(longitude_rad,x1,y1,z1)
            (r, latitude2_rad, longitude2_rad)=cartesian2spherical(x2,y2,z2)
            locations.append([radian2degree(latitude2_rad),radian2degree(longitude2_rad)])
    locations=check_borders(locations)
    folium.Polygon(
        locations=locations,
        weight=0,
        fill_color=color,
        fill_opacity=0.3,
        fill=False,
        tooltip=str(tmax)+"min"
    ).add_to(mapa)

mapa = folium.Map(location=coordinates_origin, zoom_start=4)
# draw origin
folium.CircleMarker(location=coordinates_origin,
                        radius=2,
                        color="#7aff33",
                        weight=5).add_to(mapa)

draw_parallels(mapa)
draw_meridians(mapa)
# draw_distances_to_origin(degree2radian(coordinates_origin[0]) ,degree2radian(coordinates_origin[1]), mapa)

colors=['red','blue','yellow','cyan','magenta']
mapa = folium.Map(location=coordinates_origin, zoom_start=4)
for i in range(len(perimeters)-1):
    
    print_shape(perimeters[i], perimeters[i+1], mapa, colors[i%len(colors)],degree2radian(coordinates_origin[0]) ,degree2radian(coordinates_origin[1]),)
    mapa.save("map"+str(i)+".html")

# for perimeter in perimeters[::-1]:
#     locations=[]
#     for airport in perimeter:
#         coordinates_destination=[float(data[airport]['latitude']), float(data[airport]['longitude'])]
#         folium.CircleMarker(location=coordinates_destination,
#                               tooltip=airport,
#                               radius=1,
#                               weight=5).add_to(mapa)
#         locations.append(coordinates_destination)
#         t=distance_from_airport[airport]*15  
#     folium.Polygon(
#         locations=locations,
#         color='red', 
#         weight=1,
#         fill_color='red',
#         fill_opacity=0.5,
#         fill=False,
#         tooltip=str(t)+"min"
#     ).add_to(mapa)


       
# mapa.save("map.html")