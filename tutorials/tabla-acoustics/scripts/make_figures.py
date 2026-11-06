"""Reproduce all figures for the tabla acoustics tutorial (idealized models)."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jn_zeros
from scipy.linalg import eigh
P=Path(__file__).resolve().parents[1]/'figures';P.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
# Harmonic comparison
f1=277.1826;f2=415.3047
fig,ax=plt.subplots(2,1,figsize=(10,5.5),sharex=True)
for a,f,label in zip(ax,[f1,f2],['C sharp 4 (277.18 Hz)','G sharp 4 (415.30 Hz)']):
 n=np.arange(1,10); x=n*f; y=1/n
 a.vlines(x,0,y,color='#2563a6',lw=2);a.scatter(x,y,c='#2563a6',s=15)
 for k,xx,yy in zip(n,x,y):
  if xx<2600:a.annotate(str(k),(xx,yy),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
 a.set_ylabel('Relative amplitude');a.set_ylim(0,1.16);a.set_title(label,loc='left')
 a.axvspan(818,842,color='#e7a84a',alpha=.22)
ax[-1].set_xlabel('Frequency (Hz)');ax[-1].set_xlim(0,2600)
fig.suptitle('Illustrative harmonic spectra (peak labels = harmonic order)');fig.tight_layout();fig.savefig(P/'harmonic_spectra.pdf');plt.close(fig)
# Modal shape images
x=np.linspace(-1,1,301);X,Y=np.meshgrid(x,x);r=np.hypot(X,Y);theta=np.arctan2(Y,X)
fig,axes=plt.subplots(1,4,figsize=(11.5,3.2))
for a,(m,n) in zip(axes,[(0,1),(1,1),(2,1),(0,2)]):
 z=jn_zeros(m,n)[-1]; val=jv(m,z*r)*np.cos(m*theta);val=np.ma.masked_where(r>1,val)
 a.contourf(X,Y,val,levels=27,cmap='RdBu_r');a.contour(X,Y,val,levels=[0],colors='black',linewidths=.7)
 a.set_title(f'Mode ({m},{n})\nratio {z/jn_zeros(0,1)[0]:.3f}');a.set_aspect('equal');a.axis('off')
fig.suptitle('Uniform circular membrane: red/blue are opposite displacement',y=1.05)
fig.tight_layout();fig.savefig(P/'membrane_modes.pdf');plt.close(fig)
# beating
frequency_difference=abs(3*f1-2*f2);t=np.linspace(0,3.3,10000)
sig=np.cos(2*np.pi*3*f1*t)+np.cos(2*np.pi*2*f2*t)
fig,ax=plt.subplots(figsize=(10,2.8));ax.plot(t,sig,lw=.55,color='#1c5c83');ax.plot(t,2*np.abs(np.cos(np.pi*frequency_difference*t)),color='#d17031',lw=2,label='Envelope');ax.plot(t,-2*np.abs(np.cos(np.pi*frequency_difference*t)),color='#d17031',lw=2)
ax.set_xlabel('Time (s)');ax.set_ylabel('Amplitude (arb.)');ax.set_title(f'Near-common harmonics: beating at {frequency_difference:.2f} Hz');ax.set_xlim(0,3.3);fig.tight_layout();fig.savefig(P/'beating.pdf');plt.close(fig)
# finite element radially symmetric membrane, linear hat functions; axisymmetric m
# weak form: integral [r u'v' + m^2/r uv] dr = lambda integral rho(r) r uv dr
# domain r in [0,1], u(1)=0; u(0)=0 for m>0; natural at 0 for m=0.
def spectrum(m,loading=0,patch_radius=.38,N=160,num=3):
 nodes=np.linspace(0,1,N+1);keep=np.arange(N) if m==0 else np.arange(1,N)
 lookup={v:k for k,v in enumerate(keep)};K=np.zeros((len(keep),len(keep)));M=K.copy()
 # 3 point Gauss on each element, exact-ish quadrature for jump location via subdividing at patch border
 gp=np.array([-.7745966692,0,.7745966692]);gw=np.array([5/9,8/9,5/9])
 for el in range(N):
  a,b=nodes[el],nodes[el+1]; cuts=[a]+([patch_radius] if a<patch_radius<b else [])+[b]
  for aa,bb in zip(cuts[:-1],cuts[1:]):
   if bb<=aa:continue
   rs=(aa+bb)/2+(bb-aa)*gp/2;weights=gw*(bb-aa)/2
   h=b-a;v=np.vstack(((b-rs)/h,(rs-a)/h)).T;d=np.array([-1/h,1/h]);rho=1+loading*(rs<patch_radius)
   kl=np.sum(weights*rs)*np.outer(d,d)
   ml=np.einsum('i,ij,ik->jk',weights*rs*rho,v,v)
   if m:kl+=m*m*np.einsum('i,ij,ik->jk',weights/rs,v,v)
   for i,ii in enumerate([el,el+1]):
    if ii not in lookup:continue
    for j,jj in enumerate([el,el+1]):
     if jj not in lookup:continue
     K[lookup[ii],lookup[jj]]+=kl[i,j];M[lookup[ii],lookup[jj]]+=ml[i,j]
 vals=eigh(K,M,eigvals_only=True,subset_by_index=[0,num-1]);return np.sqrt(vals)
# calculate for increasing loaded density and plot 6 key modes
loads=[0,1,2,3,4,5]
mode_names=[(0,1),(1,1),(2,1),(0,2),(3,1),(1,2)]
arr=[]
for q in loads:
 s={m:spectrum(m,q,num=2 if m in (0,1) else 1) for m in (0,1,2,3)}
 vals=np.array([s[m][n-1] for m,n in mode_names]);arr.append(vals/vals[0])
arr=np.array(arr)
fig,ax=plt.subplots(figsize=(9,4.5))
for i,(m,n) in enumerate(mode_names):ax.plot(loads,arr[:,i],marker='o',label=f'({m},{n})')
for h in [1,2,3,4]:ax.axhline(h,color='gray',lw=.7,ls='--',alpha=.5)
ax.set_xlabel('Extra central density / background density');ax.set_ylabel('Frequency / lowest mode');ax.set_title('Illustrative radial finite-element model (patch radius = 0.38 R)');ax.legend(ncol=3);fig.tight_layout();fig.savefig(P/'loading_sweep.pdf');plt.close(fig)
print('Uniform FE ratios:',np.round(arr[0],3))
print('Loaded FE ratios:',np.round(arr[2],3))
