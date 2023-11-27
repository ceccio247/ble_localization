import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import scipy
import scipy.interpolate
import math


AIRTAG_MAC = "40:22:D8:3C:1F:72"
NUM_FIGS = 2


def str_to_float(number):
    return round(float(number), NUM_FIGS)

def get_pos_loc(loc_filename):
    x = y = z = None
    lines = []
    with open(loc_filename) as f:
        reader = csv.DictReader(f)
        for line in reader:
            lines.append(line)
    x = str_to_float(lines[-1]['x']) - str_to_float(lines[0]['x'])
    y = str_to_float(lines[-1]['y']) - str_to_float(lines[0]['y'])
    z = str_to_float(lines[-1]['z']) - str_to_float(lines[0]['z'])
    return (x,y,z)

def get_rssi_vals(data_filename, loc_filename, target_mac):

    data_points = {
                'x': [],
                'y': [],
                'z': [],
                'rssi': [],
            }

    x0 = y0 = z0 = None
    with open(loc_filename) as f:
        reader = csv.DictReader(f)
        line = reader.__next__()
        x0 = str_to_float(line['x'])
        y0 = str_to_float(line['y'])
        z0 = str_to_float(line['z'])

    with open(data_filename) as f:
        reader = csv.DictReader(f)
        for line in reader:
            if line['validPos'] == 'true':
                if line['addr'] == target_mac:
                    x = str_to_float(line['x'])-x0
                    y = str_to_float(line['y'])-y0
                    z = str_to_float(line['z'])-z0

                    rssi = int(line['rssi'])

                    data_points['x'].append(x)
                    data_points['y'].append(y)
                    data_points['z'].append(z)
                    data_points['rssi'].append(rssi)

    df = pd.DataFrame({
            'x': data_points['x'],
            'y': data_points['y'],
            'z': data_points['z'],
            'rssi': data_points['rssi'],
        })

    return df




def interpolate(df, extension=0, mth='linear'):
    # define a regular grid
    step = pow(10.0,(-1*NUM_FIGS))

    GRID_DENSITY = 2000j
    grid_x, grid_z = np.mgrid[df.x.min()-extension:df.x.max()+extension:GRID_DENSITY,
            df.z.min()-extension:df.z.max()+extension:GRID_DENSITY]

    grid = scipy.interpolate.griddata(df[['x','z']], df['rssi'], (grid_x, grid_z), method=mth)

    return (grid_x, grid_z, grid)

def plot_interpolation(df, grid, label=None):
    plt.imshow(grid.T, extent=(df.x.min(), df.x.max(), df.z.min(), df.z.max()), origin='lower')
    plt.colorbar()
    plt.xlabel("x (meters)")
    plt.ylabel("z (meters)")
    plt.show()
    if label:
        plt.savefig(label)

if __name__ == '__main__':
    pos_list = ['pos1', 'pos2', 'pos3']

    for pos in pos_list:
        (xt, yt, zt) = get_pos_loc(pos+'-loc.csv')

        df = get_rssi_vals(pos+'-test-data.csv', pos+'-test-loc.csv',AIRTAG_MAC)
        (grid_x, grid_z, grid) = interpolate(df)
        ind = np.unravel_index(np.nanargmax(grid, axis=None), grid.shape)

        predicted_x = grid_x[ind[0]][0]
        predicted_z = grid_z[0][ind[1]]

        loc_err = np.sqrt((predicted_x - xt)**2 + (predicted_z - zt)**2)
        print("Error for " + pos + ": " + str(loc_err) + " meters")

        #plot_interpolation(df, grid)

