import numpy as np
from PIL import Image, ImageDraw, ImageFont
rng=np.random.default_rng(0)
exec(open('/tmp/rim_demo.py').read().split("angles=[")[0])
S_LR=192; S_HR=384
angles=[-0.5+0.09*k for k in range(12)]
views=[render(col,a,S_LR,noise=0.03) for a in angles]
gt,mgt=render(col,0.0,S_HR)

# ---- 1. 平面累积（2D 平移对齐） ----
flat=np.zeros((S_LR,S_LR,3)); n=0
for img,m in views:
    ys,xs=np.where(m)
    dx=int(round(xs.mean()-S_LR/2)); dy=int(round(ys.mean()-S_LR/2))
    flat+=np.roll(np.roll(img,-dy,0),-dx,1); n+=1
flat/=n
flat_up=np.array(Image.fromarray((flat*255).astype(np.uint8)).resize((S_HR,S_HR),Image.BICUBIC)).astype(np.float32)/255

# ---- 2. 纹理空间累积（已知曲面参数化，作为"曲边分支"） ----
TW,TH=768,384
acc=np.zeros((TH,TW,3)); wacc=np.zeros((TH,TW))
for (img,m),a in zip(views,angles):
    px,py,vis=project(a,S_LR)
    x0=np.floor(px).astype(int); y0=np.floor(py).astype(int)
    ok=(x0>=0)&(x0<S_LR)&(y0>=0)&(y0<S_LR)
    cols=img[y0[ok],x0[ok]]
    w=np.clip(np.cos(th[vis][ok]-a),0,1)**0.5
    u=((th[vis][ok]+np.pi)/(2*np.pi)*(TW-1)).astype(int)
    v=((h[vis][ok]+1)/2*(TH-1)).astype(int)
    idx=v*TW+u
    for ch in range(3):
        acc[...,ch]+=np.bincount(idx,weights=cols[:,ch]*w,minlength=TH*TW).reshape(TH,TW)
    wacc+=np.bincount(idx,weights=w,minlength=TH*TW).reshape(TH,TW)
tex=acc/np.maximum(wacc,1e-6)[...,None]
vh=np.clip(((h+1)/2*(TH-1)).astype(int),0,TH-1); uh=np.clip(((th+np.pi)/(2*np.pi)*(TW-1)).astype(int),0,TW-1)
tex_render,mtex=render(tex[vh,uh],0.0,S_HR)

# ---- 3. 压缩判据（det J）→ 路由 ----
a0=angles[3]
f0,m0=render(col,a0,S_LR,noise=0.02); f1,m1=render(col,a0+0.09,S_LR,noise=0.02)
B=8; R=5
g0=f0.mean(-1); g1=f1.mean(-1)
u_=np.zeros((S_LR//B,S_LR//B)); v_=np.zeros_like(u_)
for by in range(S_LR//B):
    for bx in range(S_LR//B):
        y0_,x0_=by*B,bx*B; ref=g0[y0_:y0_+B,x0_:x0_+B]; best=(1e9,0,0)
        for dy in range(-R,R+1):
            for dx in range(-R,R+1):
                yy,xx=y0_+dy,x0_+dx
                if yy<0 or xx<0 or yy+B>S_LR or xx+B>S_LR: continue
                e=((g1[yy:yy+B,xx:xx+B]-ref)**2).mean()
                if e<best[0]: best=(e,dy,dx)
        v_[by,bx],u_[by,bx]=best[1],best[2]
det=(1+np.gradient(u_,axis=1))*(1+np.gradient(v_,axis=0))-np.gradient(u_,axis=0)*np.gradient(v_,axis=1)
det=np.kron(det,np.ones((B,B)))[:S_LR,:S_LR]
rim=(det<0.8)
rim=np.kron(rim[::B,::B],np.ones((B*S_HR//S_LR,B*S_HR//S_LR)))[:S_HR,:S_HR]
from numpy import ones
def blur(a,k=9):
    ker=ones(k)/k
    return np.apply_along_axis(lambda r: np.convolve(r,ker,'same'),0,
           np.apply_along_axis(lambda c: np.convolve(c,ker,'same'),1,a.astype(float)))
mask=np.clip(blur(rim.astype(float),15),0,1)[...,None]
hyb=np.clip(mask*tex_render+(1-mask)*flat_up,0,1)

def ps(a,msk=mgt):
    e=((a-gt)**2*msk[...,None]).sum()/max(msk.sum()*3,1); return 10*np.log10(1/max(e,1e-12))
edge=np.zeros_like(mgt)
for y in range(S_HR):
    xs=np.where(mgt[y])[0]
    if len(xs)>4: edge[y,xs[:10]]=True; edge[y,xs[-10:]]=True
inter=mgt&~edge
print("全图:   平面 %.2f | 纹理 %.2f | 混合 %.2f"%(ps(flat_up),ps(tex_render),ps(hyb)))
print("边缘带: 平面 %.2f | 纹理 %.2f | 混合 %.2f"%(ps(flat_up,edge),ps(tex_render,edge),ps(hyb,edge)))
print("内部:   平面 %.2f | 纹理 %.2f | 混合 %.2f"%(ps(flat_up,inter),ps(tex_render,inter),ps(hyb,inter)))
canv=Image.new('RGB',(S_HR*4+24,S_HR+36),(18,18,18)); ft=ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc',18)
for k,(im,lab) in enumerate(((Image.fromarray((flat_up*255).astype(np.uint8)),"平面累积"),
                             (Image.fromarray((tex_render*255).astype(np.uint8)),"纹理累积"),
                             (Image.fromarray((hyb*255).astype(np.uint8)),"混合路由"),
                             (Image.fromarray((gt*255).astype(np.uint8)),"真实"))):
    canv.paste(im,(k*(S_HR+8),36)); ImageDraw.Draw(canv).text((k*(S_HR+8)+4,8),lab,font=ft,fill=(255,255,255))
canv.save('/tmp/hybrid_result.png'); print("saved")
