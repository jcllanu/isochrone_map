# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:27:01 2024

@author: llama
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
import math
import geometrical_utilities as geom
import utilities as util

        
norm = colors.Normalize(vmin=0, vmax=600)
f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('magma'))


# Draw paralleles in map
def draw_parallels(mapa):
    """
    Adds lines of latitude (parallels) to the given map object at regular intervals.
    Args:
        mapa (folium.Map): The folium map object to which the parallels will be added.
    """
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
    
# Draw meridians in map 
def draw_meridians(mapa):
    """
    Adds lines of longitude (meridians) to the given map object at regular intervals.
    Args:
        mapa (folium.Map): The folium map object to which the meridians will be added.
    """
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
def draw_distances_to_origin(origin_coordinates_deg, mapa):
    """
    Draws concentric lines (similar to parallels) that represent equal distances 
    from a specified origin point  on a map.
    Args:
        origin_coordinates_deg (float, float): Latitude and longitud of the origin airport in degrees
        mapa (folium.Map): The folium map object to which the lines will be added.
    """
    for i in range(10):
        x=math.cos(geom.degree2radian(i*10))
        coor=[]
        for angle in np.arange(0,2*math.pi,0.005):
            # Calculate Cartesian coordinates for the point on the sphere
            (x,y,z)=(x,(1-x**2)*math.cos(angle),(1-x**2)*math.sin(angle))
            # Rotate the point so that the image of the origin is the (latitude_rad, longitude_rad) point
            (x1,y1,z1)=geom.rot_fix_y(geom.degree2radian(origin_coordinates_deg[0]),x,y,z)
            (x2,y2,z2)=geom.rot_fix_z(geom.degree2radian(origin_coordinates_deg[1]),x1,y1,z1)
            # Convert rotated Cartesian coordinates to spherical coordinates
            (r, latitude2_rad, longitude2_rad)=geom.cartesian2spherical(x2,y2,z2)
            # Append the latitude and longitude (in degrees) to the coordinate list
            coor.append([geom.radian2degree(latitude2_rad),geom.radian2degree(longitude2_rad)])
            
            # Add a marker at each point to visualize the line
            folium.CircleMarker(
            location=[geom.radian2degree(latitude2_rad),geom.radian2degree(longitude2_rad)],
            color="red",
            tooltip=str(i),
            radius=1,
            weight=2,
            ).add_to(mapa)

# Splits a set of line segments into stretches that do not cross longitude borders (180 to -180 or viceversa).
def check_borders_line(points):
    """
    Identifies and separates stretches of a polyline that cross the 180 longitude border.
    Args:
        points (list): List of points (latitude, longitude) representing the polyline.
    Returns:
        list: A list of stretches (sub-lists of points), where no stretch crosses the border.
    """
    stretches=[] # List to store the separated stretches
    stretch=[points[0]]  # Initialize the first stretch with the starting point
    for i in range(len(points)-1):       
        origin=points[i]
        end=points[i+1]
        # Check if the segment crosses the border at latitude ±180
        if (origin[1]>160 and end[1]<-160) or (origin[1]<-160 and end[1]>160):
            stretches.append(stretch) # Save the current stretch
            stretch=[end] # Start a new stretch
        else:
            stretch.append(end) # Add point to the current stretch
    stretches.append(stretch) # Add the final stretch
    return stretches

# Draws flight routes from an origin airport on mapa.
def draw_flights_from_origin(origin, data, mapa):
    """
    Plots flight routes from an origin airport to its destinations on a map.
    Args:
        origin (str): The IATA code of the origin airport.
        data (dict): Dictionary containing airport and route data.
        mapa: A Folium map object where the routes will be drawn.
    """
    # Get the geographic coordinates of the origin airport
    coordinates_origin_deg=[float(data[origin]['latitude']),float(data[origin]['longitude'])]
    for destination in data[origin]['routes']:
        points=[] # Stores the points defining the route
        coordinates_destination_deg=[float(data[destination['iata']]['latitude']), float(data[destination['iata']]['longitude'])]
        # Convert destination coordinates to Cartesian and apply rotation that sends the origin airpot to (1,0,0)
        (x,y,z)=geom.spherical2cartesian(1, geom.degree2radian(coordinates_destination_deg[0]), geom.degree2radian(coordinates_destination_deg[1]))
        (x1,y1,z1)=geom.rot_fix_z(-geom.degree2radian(coordinates_origin_deg[1]),x,y,z)
        (x2,y2,z2)=geom.rot_fix_y(-geom.degree2radian(coordinates_origin_deg[0]),x1,y1,z1)
        (u,v)=geom.stereographic_projection(x2,y2,z2)
        # Generate points for the flight route
        N=1000
        for t in range(N+1):
            u_t = t * u/N
            v_t = t * v/N
            # Apply the inverse of the previous transformation
            (x,y,z) = geom.inverse_stereographic_projection(u_t, v_t)
            (x1,y1,z1)=geom.rot_fix_y(geom.degree2radian(coordinates_origin_deg[0]),x,y,z)
            (x2,y2,z2)=geom.rot_fix_z(geom.degree2radian(coordinates_origin_deg[1]),x1,y1,z1)
            (r, latitude2_rad, longitude2_rad)=geom.cartesian2spherical(x2,y2,z2)
            points.append([geom.radian2degree(latitude2_rad),geom.radian2degree(longitude2_rad)])
        # Separate the points into stretches that do not cross borders
        stretches=check_borders_line(points)
        hours, minutes= util.min2hours(destination['min'])
        # Add each stretch as a polyline on the map
        for stretch in stretches:
            folium.PolyLine(
            locations = stretch,
            color="#FF0000",
            weight=5,
            tooltip="From "+data[origin]['city_name']+" to "+data[destination['iata']]['city_name'] + " in "+ util.hours_minutes_printer(hours,minutes)
            ).add_to(mapa)

# Draws the IATA codes of airports on a map.        
def draw_iata_codes(origin, data, mapa):
    """
    Draws markers for the origin airport and its destination airports with their IATA codes.
    Args:
        origin (str): The IATA code of the origin airport.
        data (dict): Dictionary containing airport and route data.
        mapa: A Folium map object where the markers will be drawn.
    """
    # Get the geographic coordinates of the origin airport
    coordinates_origin_deg=[float(data[origin]['latitude']),float(data[origin]['longitude'])]
    # Draw the origin airport marker
    folium.CircleMarker(location=coordinates_origin_deg,
                          tooltip=origin,
                          radius=1,
                          color='black',
                          weight=10).add_to(mapa)
    
    # Draw markers for each destination airport
    for destination in data[origin]['routes']:
        coordinates_destination_deg=[float(data[destination['iata']]['latitude']), float(data[destination['iata']]['longitude'])]
        folium.CircleMarker(location=coordinates_destination_deg,
                              tooltip=destination['iata'],
                              radius=1,
                              color='black',
                              weight=2).add_to(mapa)

# Corrects polygon or route locations that cross specific borders.
def check_borders(locations):
    """
    Adjusts locations to handle cases where points cross the longitude borders (-180 or 180 degrees).
    Args:
        locations (list): List of points (latitude, longitude) representing a polygon or route.
    Returns:
        list: Corrected list of locations, ensuring continuity across longitude boundaries.
    """
    locations_corrected=[] # List to store corrected locations
    for i in range(len(locations)):       
        origin=locations[i]
        end=locations[(i+1)%len(locations)]
        locations_corrected.append(origin) # Add the current point to the corrected list

        # Handle case where crossing from >160 to <-160 longitude
        if origin[1]>160 and end[1]<-160:
            previous=locations[(i-1)%len(locations)]
            following=locations[(i+2)%len(locations)]
            # Project origin point to longitude 180 using the origin and previous points
            locations_corrected.append([previous[0]+(179.99999-previous[1])*(origin[0]-previous[0])/(origin[1]-previous[1]), 179.99999])
            # Move to the North Pole in the 180 meridian
            locations_corrected.append([89.99999, 179.99999])
            # Cross the 180 meridian safely
            locations_corrected.append([89.99999, -179.99999])
            # Move down from the North Pole to the projection of the end point to longitude -180 using the end and following points
            locations_corrected.append([end[0]+(-179.99999-end[1])*(following[0]-end[0])/(following[1]-end[1]), -179.99999])
        
        # Handle case where crossing from <-160 to >160 longitude   
        if origin[1]<-160 and end[1]>160:
            previous=locations[(i-1)%len(locations)]
            following=locations[(i+2)%len(locations)]
            # Project origin point to longitude 180 using the origin and previous points
            locations_corrected.append([previous[0]+(-179.99999-previous[1])*(origin[0]-previous[0])/(origin[1]-previous[1]), -179.99999])           
            # Move to the North Pole in the -180 meridian
            locations_corrected.append([89.99999, -179.99999])
            # Cross the -180 meridian safely
            locations_corrected.append([89.99999, 179.99999])
            # Move down from the North Pole to the projection of the end point to longitude 180 using the end and following points
            locations_corrected.append([end[0]+(179.99999-end[1])*(following[0]-end[0])/(following[1]-end[1]), 179.99999])
            
    return locations_corrected
            
# Plots a polygon representing the shape between airport perimeters.        
def plot_shape(inner_perimeter, outter_perimeter, distance_from_airport, XY_plane, mapa, color, origin_coordinates_deg):
    """
    Draws a shape representing an airport region on a map.
    Args:
        inner_perimeter (list): IATA codes defining the inner boundary of the shape.
        outer_perimeter (list): IATA codes defining the outer boundary of the shape.
        mapa: A Folium map object where the shape will be drawn.
        color (str): Color used to fill the shape.
        origin_coordinates_deg (float, float): Latitude and longitud of the origin airport in degrees
    """
    # Combine the outer and reversed inner perimeters to define the polygon
    perimeter = outter_perimeter + [outter_perimeter[0]] + inner_perimeter[::-1]
    
    locations=[] # List to store the interpolated polygon locations   
    for i in range(len(perimeter)):
        # For each consecutive pair of points in the combined perimeter
        airport_origin=perimeter[i]
        airport_end=perimeter[(i+1)%len(perimeter)]
        # Get the projected coordinates of the airports
        u1, v1, u2, v2 = XY_plane[airport_origin][0], XY_plane[airport_origin][1], XY_plane[airport_end][0], XY_plane[airport_end][1]
        N=50 # Number of interpolation steps
        for t in range(N+1):
            u_t = u1 + t * (u2-u1)/N
            v_t = v1 + t * (v2-v1)/N
            
            # Apply the inverse transformation that sends the projected coordinates to the sphere
            (x,y,z) = geom.inverse_stereographic_projection(u_t, v_t)
            # Apply the rotation that sends (1,0,0) to the origin airport
            (x1,y1,z1)=geom.rot_fix_y(geom.degree2radian(origin_coordinates_deg[0]),x,y,z)
            (x2,y2,z2)=geom.rot_fix_z(geom.degree2radian(origin_coordinates_deg[1]),x1,y1,z1)
            # Transform to spherical and to degrees
            (r, latitude2_rad, longitude2_rad)=geom.cartesian2spherical(x2,y2,z2)
            locations.append([geom.radian2degree(latitude2_rad),geom.radian2degree(longitude2_rad)])
            
    # Correct the locations for any border crossing issues
    locations=check_borders(locations)
    # Convert maximum time to hours and minutes
    hours, minutes= util.min2hours(max([distance_from_airport[airport]*TIME_INTERVAL for airport in outter_perimeter]))
    # Add the polygon to the map
    folium.Polygon(
        locations=locations,
        weight=0,
        fill_color=color,
        fill_opacity=0.3,
        fill=False,
        tooltip="< " + util.hours_minutes_printer(hours,minutes)
    ).add_to(mapa)

# Groups destinations by travel time intervals from a given origin airport
def group_destinations_by_time(data, origin, time_interval):
    """
    Groups destinations based on the travel time intervals from the origin airport.
    Args:
        data (dict): A dictionary containing airport data, including routes and travel times.
        origin (str): The IATA code of the origin airport.
        time_interval (int): The time interval (in minutes) to group the destinations by.
    Returns:
        tuple: A tuple containing two dictionaries:
            - distances: A dictionary where keys are the travel time intervals and values are lists of destination IATA codes.
            - distance_from_airport: A dictionary where keys are destination IATA codes and values are the corresponding travel time interval.
    """
    distances = {}  # Dictionary to store destinations grouped by time intervals
    distance_from_airport = {}  # Dictionary to store the time interval for each destination

    # Loop through the routes from the origin airport
    for destination in data[origin]['routes']:
        # Calculate the travel time interval for the destination
        d = destination['min'] // time_interval
        distance_from_airport[destination['iata']] = d  # Store the time interval for the destination

        # Group the destination by its time interval
        if d in distances:
            distances[d].append(destination['iata'])  # Add destination to the existing time interval
        else:
            distances[d] = [destination['iata']]  # Create a new group for the time interval

    return distances, distance_from_airport

# Calculates projected coordinates for airports based on their spherical coordinates.
def get_projected_coordinates(data, origin):
    """
    Projects airport locations onto a 2D plane relative to the origin airport.
    Args:
        data (dict): A dictionary containing airport data, including routes, latitudes, and longitudes.
        origin (str): The IATA code of the origin airport.
        
    Returns:
        tuple: A tuple containing two dictionaries:
            - XY_plane: A dictionary mapping each airport's IATA code to its projected (u, v) coordinates.
            - airport_from_coors: A dictionary mapping each (u, v) coordinate pair to the corresponding airport's IATA code.
    """
    UV_plane = {} # Dictionary to store the projected (u, v) coordinates for each airport
    UV_plane[origin] = (0,0) # The origin airport is at the center (0, 0)
    airport_from_coors = {} # Dictionary to map (u, v) coordinates to airports
    airport_from_coors[(0,0)] = origin # The origin airport's coordinates are (0, 0)
    # Loop through all the routes from the origin airport
    for destination in data[origin]['routes']:
        # Get the latitude and longitude of the destination airport
        coordinates_destination=[float(data[destination['iata']]['latitude']),float(data[destination['iata']]['longitude'])]       
        # Convert the destination's spherical coordinates (latitude, longitude) to Cartesian coordinates (x, y, z)
        (x,y,z)=geom.spherical2cartesian(1, geom.degree2radian(coordinates_destination[0]), geom.degree2radian(coordinates_destination[1]))
        # Apply rotation to align the origin with (x, y, z) = (1, 0, 0)
        (x1,y1,z1)=geom.rot_fix_z(-geom.degree2radian(coordinates_origin[1]),x,y,z)
        (x2,y2,z2)=geom.rot_fix_y(-geom.degree2radian(coordinates_origin[0]),x1,y1,z1)
        # Apply stereographic projection to the rotated coordinates to get the 2D (u, v) coordinates
        (u,v)=geom.stereographic_projection(x2,y2,z2)
        UV_plane[destination['iata']]=(u,v) # Store the (u, v) coordinates for the destination airport
        airport_from_coors[(u,v)]=destination['iata'] # Map the (u, v) coordinates to the destination airport's IATA code
            
    return  UV_plane, airport_from_coors    
      
def get_perimeters(origin, distances, distance_from_airport, UV_plane, airport_from_coors):
    perimeters=[] # List to store the calculated perimeters
    polygon=[] # To store the current polygon for checking if the origin is inside and the intersections    
    
    # Sort distances in ascending order
    sorted_distances = dict(sorted(distances.items(), key=lambda item: item[0]))
    for same_distance in sorted_distances:
        angles={}
        # Calculate angles for each airport at this distance level
        for airport in distances[same_distance]:
            (rho,alpha)=geom.cartesian2polar(UV_plane[airport][0],UV_plane[airport][1])
            angles[airport]=alpha
            
        # Sort airports by angle
        sorted_airports = list(dict(sorted(angles.items(), key=lambda item: item[1])).keys())
        sorted_airports.append(sorted_airports[0]) # Close the polygon loop
        
        # Extract u and v coordinates
        x_coords=[UV_plane[airport][0] for airport in sorted_airports]
        y_coords=[UV_plane[airport][1] for airport in sorted_airports] 
        
        # For the first perimeter, start a new polygon with the sorted airports                   
        if len(perimeters)==0:
            polygon=geom.Polygon([UV_plane[airport] for airport in sorted_airports])
            
            # Check if the origin airport is inside the polygon
            if polygon.is_point_inside([0,0]):
                perimeters.append(sorted_airports) # Add the polygon if origin is inside
            else:
                # Handle the case where origin is not inside the polygon 
                sorted_airports.append(origin) # Add the origin to the list
                distance_from_airport[origin]=distance_from_airport[sorted_airports[0]]
                
                # Recalculate centroid and adjust coordinates so that centroid is the new origin
                x_coords.append(0)
                y_coords.append(0)
                centroid=[sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)]
                
                # Change coordinates to use the centroid as origin
                angles={}
                for airport in sorted_airports:
                    (rho,alpha)=geom.cartesian2polar(UV_plane[airport][0]-centroid[0], UV_plane[airport][1]-centroid[1])
                    angles[airport]=alpha
    
                # Sort airports by angle around the centroid
                airports_sorted = dict(sorted(angles.items(), key=lambda item: item[1]))
                sorted_airports=list(airports_sorted.keys())
                sorted_airports.append(sorted_airports[0])
                
                # Rebuild polygon with sorted airports
                polygon=geom.Polygon([UV_plane[airport] for airport in sorted_airports])
                perimeters.append(sorted_airports)
        
        else: # Process additional perimeters
            sorted_airports = util.unique_values(sorted_airports) # Remove duplicate airport
            sorted_airports_outside_polygon=[]
            # Extract the airports outside the polygon
            for airport in sorted_airports:
                if not polygon.is_point_inside(UV_plane[airport]):
                    sorted_airports_outside_polygon.append(airport)
            
            if len(sorted_airports_outside_polygon)>0:
                sorted_airports=sorted_airports_outside_polygon+[sorted_airports_outside_polygon[0]] # Close the polygon loop
           
            # Build the new perimeter
            new_perimeter=[]
            for airport in sorted_airports:                
                if len(new_perimeter)==0:
                    new_perimeter.append(airport) # The first airport is appended automatically
                else: # For the following ones
                    # Create the segment that goes from the last airport added to the perimeter to the new airport to include
                    segment=geom.Segment(UV_plane[new_perimeter[-1]], UV_plane[airport])
                    #Check if the segments intersects the previous perimeter
                    intersects=segment.intersectsPolygon(polygon)
                    
                    old_perimeter = util.unique_values(list(perimeters[-1])) 
                    maximum = 0
                    
                    if intersects[0]: # if it does intesect, reorder the previous perimeter so that the first vertex is the one that is the end point of the edge which intersects the segment
                        old_perimeter=old_perimeter[old_perimeter.index(airport_from_coors[intersects[1].end]):]+old_perimeter[:old_perimeter.index(airport_from_coors[intersects[1].end])]
                    while intersects[0]:
                        # while the segment that goes from the last airport added to the perimeter 
                        # to the new airport to include intersects an edge of the polygon 
                        
                        index = old_perimeter.index(airport_from_coors[intersects[1].end]) # get index of the endpoint of the edge intersected
                        if index < maximum: # we have already visited that vertex. 
                            #To avoid infinite loop we add the next vertex of the previous perimeter not yet considered
                            new_perimeter.append(old_perimeter[maximum])
                            maximum = maximum + 1
                        else: # otherwise, the endpoint of the intersected edge is added to the perimeter 
                            new_perimeter.append(airport_from_coors[intersects[1].end])
                            maximum = index + 1
                        # Try again until there are no intersections
                        segment=geom.Segment(UV_plane[new_perimeter[-1]], UV_plane[airport])
                        intersects=segment.intersectsPolygon(polygon)
                    # When there are no more intersections, append the new airport to the perimeter
                    new_perimeter.append(airport)
                    
             # Update the polygon and perimeters list        
            polygon=geom.Polygon([UV_plane[airport] for airport in new_perimeter])
            
            # Check if the origin airport is inside the polygon
            if polygon.is_point_inside([0,0]):
                perimeters.append(new_perimeter)
            else:
                # Handle the case where the new perimeter does not contain the origin 
                # (the previous perimeter is not contained in the new perimeter)
                sorted_airports = util.unique_values(new_perimeter)
                sorted_airports.append(origin)
                
                # Recalculate centroid and adjust coordinates so that centroid is the new origin
                
                x_coords.append(0)
                y_coords.append(0)
                centroid=[sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)]
                
                # Change coordinates to use the centroid as origin
                angles={}
                for airport in sorted_airports:
                    (rho,alpha)=geom.cartesian2polar(UV_plane[airport][0]-centroid[0], UV_plane[airport][1]-centroid[1])
                    angles[airport]=alpha
    
                # Sort airports by angle around the centroid
                airports_sorted = dict(sorted(angles.items(), key=lambda item: item[1]))
                sorted_airports=list(airports_sorted.keys())
                
                # Get the following and previous airports to the origin in the new perimeter
                index=sorted_airports.index(origin)
                new_perimeter_airportA = sorted_airports[(index+1)%len(sorted_airports)] 
                new_perimeter_airportB = sorted_airports[(index-1)%len(sorted_airports)]
                
                # Build the previous perimeter polygone
                old_perimeter = list(perimeters[-1])
                polygon=geom.Polygon([UV_plane[airport] for airport in old_perimeter])
                
                # Get the edge that intersects the segment new_perimeter_airportA with the origin
                segmentA = geom.Segment(UV_plane[new_perimeter_airportA],[0,0])
                intersects = segmentA.intersectsPolygon(polygon)
                old_perimeter_airportA = airport_from_coors[intersects[1].origin] # this airport will be linked to new_perimeter_airportA
                
                # Get the edge that intersects the segment new_perimeter_airportB with the origin
                segmentB = geom.Segment(UV_plane[new_perimeter_airportB],[0,0])
                intersects = segmentB.intersectsPolygon(polygon)
                old_perimeter_airport_B = airport_from_coors[intersects[1].end] # this airport will be linked to new_perimeter_airportB
                
                
                old_perimeter = util.unique_values(old_perimeter)
                new_perimeter = util.unique_values(new_perimeter)
                
                index_old_perimeter_airportA = old_perimeter.index(old_perimeter_airportA)
                index_new_perimeter_airportA = new_perimeter.index(new_perimeter_airportA)
                
                #Reorder the perimeter lists so that new_perimeter_airportA and old_perimeter_airportA are the first ones
                old_perimeter = old_perimeter[index_old_perimeter_airportA:] + old_perimeter[:index_old_perimeter_airportA]
                new_perimeter = new_perimeter[index_new_perimeter_airportA:] + new_perimeter[:index_new_perimeter_airportA]
                
                #Link both perimeters
                new_perimeter = [old_perimeter_airportA] + new_perimeter[new_perimeter.index(new_perimeter_airportA):new_perimeter.index(new_perimeter_airportB)+1] + old_perimeter[old_perimeter.index(old_perimeter_airport_B):]+[old_perimeter_airportA]
                
                #Update perimeters and polygon
                perimeters.append(new_perimeter)
                polygon=geom.Polygon([UV_plane[airport] for airport in new_perimeter])
        
        # for perimeter in perimeters[-2:]:
        #     x_coords=[UV_plane[airport][0] for airport in perimeter]
        #     y_coords=[UV_plane[airport][1] for airport in perimeter] 
        #     plt.plot(x_coords, y_coords)  
        #     for i, label in enumerate(perimeter):
        #         plt.text(x_coords[i], y_coords[i], label, fontsize=12, ha='right')
        # plt.show()
        
    return perimeters 



    
# Open and read the JSON file
with open('airline_routes.json', 'r') as file:
    data = json.load(file)

TIME_INTERVAL=15
ORIGIN_AIRPORT='FRA'

coordinates_origin=[float(data[ORIGIN_AIRPORT]['latitude']),float(data[ORIGIN_AIRPORT]['longitude'])]

mapa = folium.Map(location=coordinates_origin, zoom_start=4)
draw_parallels(mapa)
draw_meridians(mapa)
    
# draw_flights_from_origin(ORIGIN_AIRPORT, data, mapa)
# draw_distances_to_origin(coordinates_origin, mapa)

distances, distance_from_airport = group_destinations_by_time(data, ORIGIN_AIRPORT, TIME_INTERVAL)
UV_plane, airport_from_coors = get_projected_coordinates(data, ORIGIN_AIRPORT)
perimeters = get_perimeters(ORIGIN_AIRPORT, distances, distance_from_airport, UV_plane, airport_from_coors)


colors=['red','green','pink','orange','cyan','yellow','magenta','blue']
for i in range(len(perimeters)-1):
    plot_shape(perimeters[i], perimeters[i+1], distance_from_airport, UV_plane, mapa, colors[i%len(colors)], coordinates_origin)

draw_iata_codes(ORIGIN_AIRPORT, data, mapa)
mapa.save(ORIGIN_AIRPORT+".html")
