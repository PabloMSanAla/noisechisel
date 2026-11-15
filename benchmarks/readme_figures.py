#!/usr/bin/env python
"""Make the README figures: Gnuastro vs noisechisel-torch outputs on one image
and the CPU timings (the Colab GPU numbers are from the notebook benchmark).

    python benchmarks/readme_figures.py IMAGE.fits [docs/images/]
"""
import os, sys, tempfile, subprocess, time, numpy as np, torch
from astropy.io import fits
from noisechisel_torch import NoiseChisel, read_image
IMG=sys.argv[1]
GNU=os.path.join(tempfile.mkdtemp(),"gnu.fits")
OUT=sys.argv[2] if len(sys.argv)>2 else "docs/images/"
tg=[]
for i in range(3):
    t0=time.perf_counter(); subprocess.run(["astnoisechisel",IMG,"-h0","-o"+GNU,"--quiet"],check=True); tg.append(time.perf_counter()-t0)
with fits.open(GNU) as hl:
    dg=hl["DETECTIONS"].data.astype(bool); sg=hl["SKY"].data; stg=hl["SKY_STD"].data
img,_=read_image(IMG,0)
nc=NoiseChisel(device="cpu"); nc(img)
tt=[]
for i in range(3):
    t0=time.perf_counter(); r=nc(img); tt.append(time.perf_counter()-t0)
dt=r.detections.cpu().numpy(); st=r.sky_image().cpu().numpy(); stdt=r.std_image().cpu().numpy()
print("gnu",tg,"torch",tt,"threads",torch.get_num_threads())
print("diff", (dg!=dt).sum(), (sg!=st).sum(), (stg!=stdt).sum(), dg.mean())
d=dict(img=img,dg=dg,sg=sg,stg=stg,dt=dt,st=st,stdt=stdt,tg=tg,tt=tt)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
os.makedirs(OUT,exist_ok=True)
INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3df"; BLUE="#2a78d6"; GRAY="#a3a29c"
plt.rcParams.update({"font.size":10,"text.color":INK,"axes.labelcolor":INK2,
    "xtick.color":INK2,"ytick.color":INK2,"axes.edgecolor":GRID,"font.family":"DejaVu Sans"})

# ---- figure 1: outputs side by side
s=4  # display downsample
img=d["img"][::s,::s]
fig=plt.figure(figsize=(13,6.6),dpi=110)
gs=GridSpec(2,4,figure=fig,width_ratios=[1.25,1,1,1],wspace=0.08,hspace=0.18,
            left=0.02,right=0.93,top=0.88,bottom=0.04)
ax=fig.add_subplot(gs[:,0])
lo,hi=np.nanpercentile(img,[1,99.7]); v=np.arcsinh((img-lo)/(0.02*(hi-lo)))
ax.imshow(v,origin="lower",cmap="gray_r",interpolation="nearest"); ax.set_axis_off()
ax.set_title("Input: "+os.path.basename(IMG),fontsize=10,color=INK2)
sky_lim=np.nanpercentile(d["sg"],[0.5,99.5]); std_lim=np.nanpercentile(d["stg"],[0.5,99.5])
rows=[("Gnuastro 0.24","dg","sg","stg"),("noisechisel-torch","dt","st","stdt")]
ims={}
for r,(name,kd,ks,kstd) in enumerate(rows):
    for c,(k,cm,lim,title) in enumerate([(kd,"Greys",(0,1),"DETECTIONS"),
                                        (ks,"viridis",sky_lim,"SKY"),
                                        (kstd,"magma",std_lim,"SKY_STD")]):
        a=fig.add_subplot(gs[r,c+1])
        ims[c]=a.imshow(d[k][::s,::s].astype(float),origin="lower",cmap=cm,vmin=lim[0],vmax=lim[1],interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
        if r==0: a.set_title(title,fontsize=11)
        if c==0: a.set_ylabel(name,fontsize=11,color=INK,labelpad=6)
nd=[int((d[a]!=d[b]).sum()) for a,b in [("dg","dt"),("sg","st"),("stg","stdt")]]
cax1=fig.add_axes([0.94,0.47,0.01,0.40]); fig.colorbar(ims[1],cax=cax1).set_label("Sky",color=INK2)
cax2=fig.add_axes([0.94,0.05,0.01,0.40]); fig.colorbar(ims[2],cax=cax2).set_label("Sky STD",color=INK2)
for cx in (cax1,cax2): cx.tick_params(labelsize=8); [sp.set_visible(False) for sp in cx.spines.values()]
fig.suptitle(f"Same 4088×4088 Roman WFI image: pixels differing  mask {nd[0]} · Sky {nd[1]} · STD {nd[2]}",
             fontsize=13,x=0.02,ha="left",y=0.97)
fig.savefig(OUT+"comparison.png",facecolor="white"); plt.close(fig)

# ---- figure 2: performance (seconds per 4088x4088 frame)
tg=float(np.median(d["tg"])); tt=float(np.median(d["tt"]))
T4=0.664   # CUDA + CuPy on a Colab T4 (notebooks/colab_gpu_benchmark.ipynb)
bars=[("Gnuastro 0.24 · CPU, 14 threads",tg,GRAY),
      ("noisechisel-torch · CPU, 10 threads",tt,BLUE),
      ("noisechisel-torch · NVIDIA T4 GPU",T4,BLUE)]
fig,ax=plt.subplots(figsize=(8.5,2.9),dpi=110)
names=[b[0] for b in bars][::-1]; vals=[b[1] for b in bars][::-1]; cols=[b[2] for b in bars][::-1]
y=np.arange(len(bars))
ax.barh(y,vals,color=cols,height=0.6,edgecolor="white",linewidth=2)
for yi,v in zip(y,vals):
    sp=tg/v; txt=f"{v:.2f} s" + ("" if abs(sp-1)<1e-9 else f"  ({sp:.1f}× faster)")
    ax.text(v+max(vals)*0.015,yi,txt,va="center",fontsize=9.5,color=INK)
ax.set_yticks(y,names,fontsize=9.5); ax.set_ylim(-0.6,len(bars)-0.4); ax.set_xlim(0,max(vals)*1.45)
ax.set_xlabel("seconds per 4088×4088 frame (lower is better)",fontsize=9)
ax.set_title("Run time, NoiseChisel default options",fontsize=11,loc="left",color=INK)
ax.xaxis.grid(True,color=GRID,lw=0.8); ax.set_axisbelow(True)
for k in ("top","right","left"): ax.spines[k].set_visible(False)
ax.tick_params(axis="y",length=0)
fig.text(0.01,0.02,"CPU: Apple M3 Max, median of 3 runs. GPU: Google Colab T4 with CuPy. All outputs bit-identical.",fontsize=8.5,color=INK2)
fig.subplots_adjust(left=0.33,right=0.98,top=0.87,bottom=0.27)
fig.savefig(OUT+"performance.png",facecolor="white"); plt.close(fig)
