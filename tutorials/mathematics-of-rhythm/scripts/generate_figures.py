"""Reproducible illustrative rhythm figures. No measured audio is used."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path(__file__).resolve().parent.parent/'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(2,1,figsize=(10,4.8),sharex=True)
for a,N,marks,title in [(ax[0],16,[0,4,8,12],'Teentaal: 16 matras, 4+4+4+4'),(ax[1],10,[0,2,5,7],'Jhaptal: 10 matras, 2+3+2+3')]:
    ts=np.arange(2*N); amp=np.ones(2*N)*.35
    amp[np.isin(ts%N,marks)]=.85
    a.vlines(ts,0,amp,color='#1d6996',lw=1.7);a.scatter(ts,amp,color='#1d6996',s=15)
    for k in range(2):
      a.axvline(k*N,ls='--',color='#d1495b',lw=1)
    a.set_title(title,loc='left');a.set_ylim(0,1.05);a.set_ylabel('Illustrative weight')
ax[-1].set_xlabel('Matra index (two avartans)');fig.tight_layout();fig.savefig(OUT/'impulse_trains.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(9,4.5))
for ax,N,group,name in [(axs[0],16,[4,4,4,4],'Teentaal'),(axs[1],10,[2,3,2,3],'Jhaptal')]:
    th=np.pi/2-2*np.pi*np.arange(N)/N
    pts=np.column_stack([np.cos(th),np.sin(th)])
    ax.plot(np.cos(np.linspace(0,2*np.pi,400)),np.sin(np.linspace(0,2*np.pi,400)),color='.65')
    bounds=np.cumsum([0]+group[:-1]); colors=['#d1495b' if j==0 else ('#8d99ae' if j==2 else '#1d6996') for j in range(4)]
    for i,(x,y) in enumerate(pts):
      c=colors[list(bounds).index(i)] if i in bounds else '#444444'
      ax.scatter(x,y,s=140 if i in bounds else 35,c=c,zorder=3)
      ax.text(1.20*x,1.20*y,str(i+1),ha='center',va='center',fontsize=9)
    ax.set_title(name);ax.set_aspect('equal');ax.axis('off');ax.set_xlim(-1.4,1.4);ax.set_ylim(-1.4,1.4)
fig.tight_layout();fig.savefig(OUT/'cycle_diagrams.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,3.3))
a=np.ones(16)*.3; a[[0,4,8,12]]=[1,.75,.15,.7]
A=np.fft.fft(a);shift=np.roll(a,3);B=np.fft.fft(shift)
k=np.arange(16);ax.stem(k,np.abs(A),linefmt='#1d6996',markerfmt='o',basefmt=' ',label='Original')
ax.scatter(k,np.abs(B),facecolors='none',edgecolors='#d1495b',s=70,label='Shift by 3 matras')
ax.set_xlabel('DFT bin k');ax.set_ylabel('Magnitude');ax.set_title('Cyclic shifts preserve the DFT magnitude');ax.legend();fig.tight_layout();fig.savefig(OUT/'dft_shift.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,2.8)); starts=[4,8,12]
for i,s in enumerate(starts):
    ax.plot([s,s+4],[3-i,3-i],linewidth=5,solid_capstyle='round',color=['#1d6996','#4ca1a3','#e4a24a'][i]); ax.scatter([s,s+4],[3-i]*2,color='black',s=20,zorder=5)
ax.axvline(16,color='#d1495b',ls='--');ax.text(16.12,3.3,'Sam',color='#d1495b');ax.set_xticks(range(0,18,2));ax.set_yticks([1,2,3],['Phrase 3','Phrase 2','Phrase 1']);ax.set_xlim(3,17.2);ax.set_xlabel('Elapsed matras from initial sam (0-based)');ax.set_title('An illustrative tihai: phrase duration 4, repeated every 4 matras');fig.tight_layout();fig.savefig(OUT/'tihai.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,2.5))
for y,n,col in [(1,3,'#1d6996'),(0,4,'#d1495b')]:
 ts=np.arange(n+1)/n; ax.vlines(ts,y-.25,y+.25,color=col,lw=2);ax.scatter(ts,np.ones_like(ts)*y,color=col)
ax.set_yticks([0,1],['4 pulses','3 pulses']);ax.set_xlim(-.03,1.03);ax.set_xticks(np.linspace(0,1,7));ax.set_xlabel('Normalized common cycle');ax.set_title('Three-against-four polyrhythm');fig.tight_layout();fig.savefig(OUT/'polyrhythm.pdf');plt.close(fig)
print('Generated 5 figures in',OUT)
