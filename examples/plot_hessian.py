import h5py
import numpy as np
import pylab as pl


for da in ['1']:
    for D in ['1','3','5']:
        for W in ['1','2','3']:
            ax1 = pl.subplot(131)
            ax2 = pl.subplot(132)
            ax3 = pl.subplot(133)
            f = h5py.File('cifar10_'+str(da)+'_'+D+'_'+W+'.h5','r')
            loss    = list()
            accu    = list()
            hessian_t = list()
            hessian_f = list()
            labels  = list()
            for i in sorted(f['test_set/accu']):
                print(i)
                if 'descr' not in i:
                    accu.append(f['test_set/accu/'+i][...])
            for i in sorted(f['train_set/loss']):
                print(i)
                if 'descr' not in i:
                    loss.append(f['train_set/loss/'+i][...])
            for i in sorted(f['train_set/hessian']):
                print(i)
                if 'descr' not in i:
                    labels = f['train_set/hessian/'+i][...][:,-1].astype('int32')
                    hessian_t.append(f['train_set/hessian/'+i][...][:,labels].sum())
                    hessian_f.append((f['train_set/hessian/'+i][...][:,:-1].sum()\
                            -f['train_set/hessian/'+i][...][:,labels].sum())/9)
            f.close()
            ax1.plot(np.concatenate(accu),linewidth=3)
            ax2.plot(np.concatenate(loss),linewidth=3)
            ax3.plot(np.array(hessian_t),linewidth=3)
            ax3.plot(np.array(hessian_f),linewidth=3)
            ax3.plot(np.concatenate(hessian_t)*0.1+np.concatenate(hessian_f)*0.9,
                                linewidth=3)
            pl.show()
