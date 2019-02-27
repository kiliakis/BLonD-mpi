import numpy as np
import csv

path = 'results/raw/SPS-72B-4MPPB-uint16-r4-2/'
red = path.split('-r')[1].split('-')[0]

for name in ['avg-report.csv', 'comm-comp-report.csv']:
    fname = path + name
    data = np.genfromtxt(fname, delimiter='\t', dtype=str)
    header = list(data[0])
    data = data[1:]
    header = header[:7] + ['red'] + header[7:]
    
    newdata = np.array((len(data), len(data[0])+1), dtype=str)
    newdata = []
    for i in range(len(data)):
        newdata.append(list(data[i][:7]) + [red] + list(data[i][7:]))
    #  newdata[:, :7] = data[:, :7]
    #  newdata[:, 7] = np.array(len(data), red)
    #  newdata[:, 7:] = data[:, 7:]
    writer = csv.writer(open(fname,'w'), delimiter='\t')
    writer.writerow(header)
    writer.writerows(newdata)
    #print(newdata)
    #np.savetxt(fname+'.test', newdata, header=header, delimiter='\t')
