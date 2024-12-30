# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:46:18 2024

@author: llama
"""
# Converts hours and minutes into a formatted string
def hours_minutes_printer(hours,minutes):
    """
    Formats a time duration into a human-readable string representation.
    Args:
        hours (int): The number of hours.
        minutes (int): The number of minutes.
    Returns:
        str: A string in one of the formats:
             - "Xmin" if hours == 0
             - "Xh" if minutes == 0
             - "Xh Ymin" otherwise
    """
    if hours==0:
        return str(minutes)+"min"
    elif minutes==0:
        return str(hours)+"h"
    else:
        return str(hours)+"h "+str(minutes)+"min"
    
# Converts total minutes into hours and minutes
def min2hours(minutes):
    """
   Converts a time duration in minutes into hours and remaining minutes.
   Args:
       minutes (int): The total duration in minutes.
   Returns:
       tuple: A tuple (hours, remaining_minutes).
   """
    return (minutes//60, minutes%60)

# Extracts unique values from a list
def unique_values(l):
    """
   Generates a list of unique values from the input list while preserving order.
   Args:
       l (list): The input list.
   Returns:
       list: A new list containing only unique values from the input.
   """
    unique_values = []
    for value in l:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values

# Converts a floating-point value to a hexadecimal RGB color string.
def f2hex(f2rgb, f):
    """
    Converts a floating-point value to a hexadecimal color code using an f2rgb mapping.
    Args:
        f2rgb: An object with a `to_rgba` method to map f to an RGBA tuple.
        f (float): The input floating-point value.
    Returns:
        str: The hexadecimal RGB color string (e.g., "#RRGGBB").
    """
    rgb = f2rgb.to_rgba(f)[:3]
    return '#%02x%02x%02x' % tuple([int(255*fc) for fc in rgb])

