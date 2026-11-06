"""Illustrative, smooth radially graded syahi model; not a measured density profile."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'figures';P.mkdir(exist_ok=True)

def density(r,contrast=6.,radius=.42,transition=.045):
    """A smooth, syahi-like central loading: sigma(r)/sigma0."""
    return 1.+contrast/(1.+np.exp(np.clip((np.asarray(r)-radius)/transition,-500,500)))

def spectrum(m,contrast=6.,radius=.42,transition=.045,N=100,num=3):
    nodes=np.linspace(0,1,N+1)
    keep=np.arange(N) if m==0 else np.arange(1,N)
    lookup={v:k for k,v in enumerate(keep)}
    K=np.zeros((len(keep),len(keep)));M=K.copy()
    gp=np.array([-.7745966692,0,.7745966692]);gw=np.array([5/9,8/9,5/9])
    for el in range(N):
        a,b=nodes[el],nodes[el+1]; h=b-a
        rs=(a+b)/2+(b-a)*gp/2; weights=gw*(b-a)/2
        v=np.vstack(((b-rs)/h,(rs-a)/h)).T; d=np.array([-1/h,1/h])
        kl=np.sum(weights*rs)*np.outer(d,d)
        if m:kl+=m*m*np.einsum('i,ij,ik->jk',weights/rs,v,v)
        ml=np.einsum('i,ij,ik->jk',weights*rs*density(rs,contrast,radius,transition),v,v)
        for i,ii in enumerate([el,el+1]):
            if ii not in lookup:continue
            for j,jj in enumerate([el,el+1]):
                if jj not in lookup:continue
                K[lookup[ii],lookup[jj]]+=kl[i,j];M[lookup[ii],lookup[jj]]+=ml[i,j]
    return np.sqrt(eigh(K,M,eigvals_only=True,subset_by_index=[0,num-1]))

if __name__=='__main__':
    modes=[(0,1),(1,1),(2,1),(0,2),(3,1),(1,2)]
    contrasts=[0,2,4,6,8,10]
    vals=[]
    for contrast in contrasts:
        s={m:spectrum(m,contrast,num=2 if m in [0,1] else 1) for m in [0,1,2,3]}
        v=np.array([s[m][n-1] for m,n in modes]);vals.append(v/v[0])
    vals=np.asarray(vals)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    rr=np.linspace(0,1,300)
    for contrast in [0,2,6,10]:axes[0].plot(rr,density(rr,contrast),label=rf'$\Delta\sigma/\sigma_0={contrast}$')
    axes[0].set_xlabel('$r/R$');axes[0].set_ylabel(r'$\sigma(r)/\sigma_0$');axes[0].set_title('Smooth syahi-like mass profile');axes[0].legend(fontsize=8)
    for i,(m,n) in enumerate(modes):axes[1].plot(contrasts,vals[:,i],marker='o',label=f'({m},{n})')
    for h in (1,2,3):axes[1].axhline(h,color='grey',ls='--',lw=.7,alpha=.5)
    axes[1].set_xlabel(r'Central density excess $\Delta\sigma/\sigma_0$');axes[1].set_ylabel('Mode frequency / lowest mode');axes[1].set_title('Resulting mode-frequency ratios');axes[1].legend(ncol=2,fontsize=8)
    fig.suptitle('Physically motivated profile, not an experimentally measured syahi');fig.tight_layout()
    fig.savefig(P/'smooth_syahi_model.pdf',bbox_inches='tight');plt.close(fig)
    print('Mode order:',modes)
    for c,v in zip(contrasts,vals):print(c,np.round(v,3))
