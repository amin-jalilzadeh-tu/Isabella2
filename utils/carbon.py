def calculate_window_carbon(window_U_Factor):
        if window_U_Factor == 2.90:
            return 0
        elif window_U_Factor == 1.20:
            return 70
        elif window_U_Factor == 1.21:
            return 50
        elif window_U_Factor == 0.8:
            return 150
        elif window_U_Factor == 0.81:
            return 120
        else:
            raise ValueError("windows_U_Factor not valid")
        
def calculate_floor_carbon(groundfloor_thermal_resistance):
        if groundfloor_thermal_resistance == 0.41:
            return 0
        elif groundfloor_thermal_resistance == 4.8:
            return 10
        elif groundfloor_thermal_resistance == 5:
            return 5.92
        elif groundfloor_thermal_resistance == 5.5:
            return 11
        elif groundfloor_thermal_resistance == 5.6:
            return 7
        else:
            raise ValueError("groundfloor_thermal_resistance not valid")

def calculate_facade_carbon(ext_walls_thermal_resistance):
        if ext_walls_thermal_resistance == 0.45:
            return 0
        elif ext_walls_thermal_resistance == 4.2:
            return 9.36
        elif ext_walls_thermal_resistance == 4.4:
            return 4.83
        elif ext_walls_thermal_resistance == 6.5:
            return 17.16
        elif ext_walls_thermal_resistance == 6.7:
            return 8.5
        else:
            raise ValueError("ext_walls_thermal_resistance not valid")

def calculate_roof_carbon(roof_thermal_resistance):
        if roof_thermal_resistance == 0.48:
            return 0
        elif roof_thermal_resistance == 4.5:
            return 23.29
        elif roof_thermal_resistance == 4.7:
            return 4.76
        elif roof_thermal_resistance == 8.5:
            return 18.5
        elif roof_thermal_resistance == 8.7:
            return 10.68
        else:
            raise ValueError("roof_thermal_resistance not valid")
        
# Calculate total carbon
def calculate_total_carbon(window_U_Factor, groundfloor_thermal_resistance, ext_walls_thermal_resistance, roof_thermal_resistance):
        window_carbon = calculate_window_carbon(window_U_Factor=window_U_Factor)
        floor_carbon = calculate_floor_carbon(groundfloor_thermal_resistance=groundfloor_thermal_resistance)
        facade_carbon = calculate_facade_carbon(ext_walls_thermal_resistance=ext_walls_thermal_resistance)
        roof_carbon = calculate_roof_carbon(roof_thermal_resistance=roof_thermal_resistance)
        return window_carbon + floor_carbon + facade_carbon + roof_carbon

