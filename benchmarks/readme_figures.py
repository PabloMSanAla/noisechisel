#!/usr/bin/env python
"""Make the README comparison figure: Gnuastro vs noisechisel-torch outputs on
one image (the timing figures are made by ``benchmarks/scaling.py``).

    python benchmarks/readme_figures.py IMAGE.fits [docs/images/]
"""
import os, sys, tempfile, subprocess, numpy as np
from astropy.io import fits
from noisechisel_torch import NoiseChisel
IMG=sys.argv[1]
GNU=os.path.join(tempfile.mkdtemp(),"gnu.fits")
OUT=sys.argv[2] if len(sys.argv)>2 else "docs/images/"
subprocess.run(["astnoisechisel",IMG,"-h0","-o"+GNU,"--quiet"],check=True)
with fits.open(GNU) as hl:
    dg=hl["DETECTIONS"].data.astype(bool); sg=hl["SKY"].data; stg=hl["SKY_STD"].data
img=fits.getdata(IMG,0).astype(np.float32)
r=NoiseChisel(device="cpu")(img)
dt=r.detections.cpu().numpy(); st=r.sky_image().cpu().numpy(); stdt=r.std_image().cpu().numpy()
print("diff", (dg!=dt).sum(), (sg!=st).sum(), (stg!=stdt).sum(), dg.mean())
d=dict(img=img,dg=dg,sg=sg,stg=stg,dt=dt,st=st,stdt=stdt)

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
ax.set_title("Input (simulated Roman WFI)",fontsize=10,color=INK2)
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
fig.suptitle(f"Simulated 4088×4088 Roman WFI image: pixels differing  mask {nd[0]} · Sky {nd[1]} · STD {nd[2]}",
             fontsize=13,x=0.02,ha="left",y=0.97)
fig.savefig(OUT+"comparison.png",facecolor="white"); plt.close(fig)
