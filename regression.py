import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import scipy


def get_correlation_coeff(y, y_hat):
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((y_hat-ybar)**2)
    sstot = np.sum((y - ybar)**2)    

    return ssreg/sstot

def get_polynomial_predict(x, y, deg):
    coeff = np.polyfit(x, y, deg)
    y_hat = []
    for x_hat in x:
        this_result = 0
        for i in range(len(coeff)):
            r = pow(x_hat,i)*coeff[len(coeff)-1-i]
            this_result += r
        y_hat.append(this_result)
    
    print(get_correlation_coeff(y, y_hat))
    
    return y_hat

def get_exp_predict(x, y):
    epsilon = 0.00000000001
    y_for_log = [y_hat + epsilon for y_hat in y]
    coeff = np.polyfit(x, np.log(y_for_log), 1, w=np.sqrt(y_for_log))

    y_hat = []
    for x_hat in x:
        this_result = np.exp(coeff[0]*x_hat) * np.exp(coeff[1]) - epsilon
        y_hat.append(this_result)

    print(get_correlation_coeff(y, y_hat))

    return y_hat

def get_log_predict(x, y):
    x_for_log = np.abs(x)
    coeff = np.polyfit(np.log(x_for_log), y, 1)
    y_hat = []
    for x_hat in x_for_log:
        y_hat.append(coeff[0]*np.log(x_hat)+coeff[1])

    print(get_correlation_coeff(y, y_hat))
    
    return y_hat

def get_scipy_predict(x, y):
    a_guess = 2
    b_guess = -0.5
    c_guess = -10
    d_guess = 2

    popt, pcov = scipy.optimize.curve_fit(lambda t, a, b, c, d: a * np.exp(b * t + c) + d, x, y,
            p0=(a_guess, b_guess, c_guess, d_guess), maxfev=50000)
    a = popt[0]
    b = popt[1]
    c = popt[2]
    d = popt[3]

    print(popt)

    y_hat = [min(a * np.exp(b * x_hat + c) + d,10) for x_hat in x]
    print(get_correlation_coeff(y, y_hat))
    return y_hat


AIRTAG_MAC = "53:6D:2A:D7:9F:F2"

data_points = {
            'x': [],
            'y': [],
            'z': [],
            'rssi': [],
        }
with open('data.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        line = line[:len(line)-1]
        x = float(line[0])
        y = float(line[1])
        z = float(line[2])
        rssi = {}
        for i in range(3, len(line), 2):
            rssi[line[i]] = float(line[i+1])

        data_points['x'].append(x)
        data_points['y'].append(y)
        data_points['z'].append(z)
        data_points['rssi'].append(rssi)

airtag = {
        'x': data_points['x'],
        'y': data_points['y'],
        'z': data_points['z'],
        'dist': np.sqrt([sum(x) for x in zip(np.square(data_points['x']),
                                    np.square(data_points['y']),
                                    np.square(data_points['z']))]),
        'rssi': [entry[AIRTAG_MAC] for entry in data_points['rssi']],
    }


data_points = {
            'x': [],
            'y': [],
            'z': [],
            'rssi': [],
            'dist': [],
        }


for i in range(1,len(airtag['x'])):
    if airtag['rssi'][i] != airtag['rssi'][i-1]:
        data_points['x'].append(airtag['x'][i])
        data_points['y'].append(airtag['y'][i])
        data_points['z'].append(airtag['z'][i])
        data_points['rssi'].append(airtag['rssi'][i])
        data_points['dist'].append(airtag['dist'][i])

airtag_df = pd.DataFrame(data_points)
airtag_df = airtag_df.sort_values(by=['rssi'])

plt.scatter(airtag_df['rssi'], airtag_df['dist'])
plt.plot(airtag_df['rssi'], get_exp_predict(airtag_df['rssi'], airtag_df['dist']), color='red')
plt.xlabel('rssi')
plt.ylabel('distance from airtag (meters)')
plt.show()

