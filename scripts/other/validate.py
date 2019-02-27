import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import random

reduce1 = np.genfromtxt('out/coords_10000_reduce1.dat', names=('dt', 'dE')) #, max_rows=1000000)
reduce2 = np.genfromtxt('out/coords_10000_reduce2.dat', names=('dt', 'dE')) #, max_rows=1000000)
reduce5 = np.genfromtxt('out/coords_10000_reduce5.dat', names=('dt', 'dE')) #, max_rows=1000000)


sample = random.sample(range(0, len(reduce1)), 1000000)
reduce1 = reduce1[sample]
reduce2 = reduce2[sample]
reduce5 = reduce5[sample]
# reduce2 = random.sample(reduce2, 10000)
# reduce5 = random.sample(reduce5, 10000)

fig = plt.figure()
gs = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax5 = plt.subplot(gs[0, 2])
ax = plt.subplot(gs[1, :])

# plt.title('Bunch Distribution')
# plt.xlabel('dt')
# plt.ylabel('dE')

ax1.plot(reduce1['dt'], reduce1['dE'], ls='', marker='.', ms=1, label='reduce1')
# ax1.legend()
ax1.set_title('Reduce every turn')

ax2.plot(reduce2['dt'], reduce2['dE'], ls='', marker='.', ms=1, label='reduce2')
ax2.set_title('Reduce every 2 turns')
# ax2.legend()

ax5.plot(reduce5['dt'], reduce5['dE'], ls='', marker='.', ms=1, label='reduce5')
ax5.set_title('Reduce every 5 turns')
# ax5.legend()

ax.plot(reduce1['dt'], reduce1['dE'], ls='', marker='.', ms=1, label='reduce1')
ax.plot(reduce2['dt'], reduce2['dE'], ls='', marker='.', ms=1, label='reduce2')
ax.plot(reduce5['dt'], reduce5['dE'], ls='', marker='.', ms=1, label='reduce5')
ax.legend()

plt.tight_layout()
plt.savefig('out/distributions.pdf', bbox_inches='tight')
plt.show()
plt.close()


exit()

dtdiff21 = np.abs(reduce1['dt'] - reduce2['dt'])
dEdiff21 = np.abs(reduce1['dE'] - reduce2['dE'])
dtdiff51 = np.abs(reduce5['dt'] - reduce1['dt'])
dEdiff51 = np.abs(reduce5['dE'] - reduce1['dE'])


dtreldif21 = 100 * np.abs(dtdiff21 / np.maximum(reduce1['dt'], reduce2['dt']))
dEreldif21 = 100 * np.abs(dEdiff21 / np.maximum(reduce1['dE'], reduce2['dE']))

dtreldif51 = 100 * np.abs(dtdiff51 / np.maximum(reduce1['dt'], reduce5['dt']))
dEreldif51 = 100 * np.abs(dEdiff51 / np.maximum(reduce1['dE'], reduce5['dE']))


for dif, figname in zip([dtreldif21, dEreldif21, dtreldif51, dEreldif51], ['dtreldif21.pdf', 'dEreldif21.pdf', 'dtreldif51.pdf', 'dEreldif51.pdf']):
    fig = plt.figure()
    plt.title('Error Histogram')
    plt.xlabel('Error %')
    plt.ylabel('Particles %')
    # dif = dif[np.where(dif<100)[0]]
    plt.hist(dif, bins=100, range=(0,100), density=True)
    # hist = np.zeros(100)
    # for i in dif:
    #     if(i >=100):
    #         hist[99]+=1
    #     else:
    #         hist[int(np.floor(i))]+=1
    # plt.axvline(x=np.mean(dif), color='black', label='x='+str(np.mean(dif)))
    # plt.plot(np.arange(1, 101), hist)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')
    plt.show()
    plt.close()
