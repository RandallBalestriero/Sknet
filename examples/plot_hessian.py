import h5py
import numpy as np
import pylab as pl

ax1 = pl.subplot(121)
ax2 = pl.subplot(122)

for epsilon in ['0.01','0.1']:
    for gamma in ['0.0','0.1','1.0']:
        f = h5py.File('cifar10_'+epsilon+'_'+gamma+'.h5','r')
        loss = list()
        accu = list()
        hessian = list()
        for i in f['test_set/accu']:
            if 'descr' not in i:
                accu.append(f['test_set/accu/'+i][...][1::2])
        for i in f['train_set/loss']:
            if 'descr' not in i:
                loss.append(f['train_set/loss/'+i][...][:,0])
                hessian.append(f['train_set/loss/'+i][...][:,1])
        if gamma=='0.0':
            ax1.plot(np.concatenate(accu),linewidth=3)
            ax2.plot(np.concatenate(hessian),linewidth=3)
            ax2.plot(np.concatenate(loss),linewidth=3)
        else:
            ax1.plot(np.concatenate(accu))
            ax2.plot(np.concatenate(hessian))
            ax2.plot(np.concatenate(loss))
        f.close()



pl.show()
