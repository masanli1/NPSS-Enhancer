import numpy as np
from PIL import Image, ImageDraw, ImageFont
rng=np.random.default_rng(0)

# ---------- 真实纹理：Kodak 图贴到圆柱侧面 ----------
img=np.array(Image.open('/Users/ma/Downloads/降噪/图像降噪/NPSS-Best/kodim23.png').convert('RGB')).astype(np.float32)/255
TH_,TW_=img.shape[:2]
N=40000
th=rng.uniform(-np.pi,np.pi,N); h=rng.uniform(-1,1,N)
def sample(uv_u,uv_v):
    x=np.clip((uv_u*(TW_-1)).astype(int),0,TW_-1); y=np.clip((uv_v*(TH_-1)).astype(int),0,TH_-1)
    return img[y,x]
col=sample((th+np.pi)/(2*np.pi),(h+1)/2)
F=2.2; S_LR=192; S_HR=384; CAM=np.array([0,0,3.2])
def project(Rz,S):
    x=np.sin(th)*np.cos(Rz)+np.cos(th)*np.sin(Rz)
    z=-np.sin(th)*np.sin(Rz)+np.cos(th)*np.cos(Rz)
    X=np.stack([x,h,z],1)+CAM
    vis=z>0.05
    u=F*X[vis,0]/X[vis,2]; v=F*X[vis,1]/X[vis,2]
    return u*S/2+S/2, v*S/2+S/2, vis
def render(colors,Rz,S,noise=0.0):
    px,py,vis=project(Rz,S); c=colors[vis]
    if noise: c=np.clip(c+rng.normal(0,noise,c.shape),0,1)
    x0=np.floor(px).astype(int); y0=np.floor(py).astype(int); fx=px-x0; fy=py-y0
    acc=np.zeros((S,S,3)); wa=np.zeros((S,S))
    for oy in (0,1):
        for ox in (0,1):
            xx=x0+ox; yy=y0+oy
            mm=(xx>=0)&(xx<S)&(yy>=0)&(yy<S)
            w=((1-fy) if oy==0 else fy)*((1-fx) if ox==0 else fx)*mm
            idx=yy[mm]*S+xx[mm]
            for ch in range(3): acc[...,ch]+=np.bincount(idx,weights=c[mm,ch]*w[mm],minlength=S*S).reshape(S,S)
            wa+=np.bincount(idx,weights=w[mm],minlength=S*S).reshape(S,S)
    return np.clip(acc/np.maximum(wa,1e-6)[...,None],0,1), wa>1e-6
angles=[-0.5+0.09*k for k in range(12)]
views=[render(col,a,S_LR,noise=0.03) for a in angles]
gt,mgt=render(col,0.0,S_HR)
bi=np.array(Image.fromarray((views[0][0]*255).astype(np.uint8)).resize((S_HR,S_HR),Image.BICUBIC)).astype(np.float32)/255
# 平面累积（质心对齐，作为粗略基线）
flat=np.zeros((S_LR,S_LR,3)); n=0
for img_,m in views:
    ys,xs=np.where(m); dx=int(round(xs.mean()-S_LR/2)); dy=int(round(ys.mean()-S_LR/2))
    flat+=np.roll(np.roll(img_,-dy,0),-dx,1); n+=1
flat/=n
flat_up=np.array(Image.fromarray((flat*255).astype(np.uint8)).resize((S_HR,S_HR),Image.BICUBIC)).astype(np.float32)/255
# 纹理空间累积
TWt,THt=768,384
acc=np.zeros((THt,TWt,3)); wacc=np.zeros((THt,TWt))
for (im,m),a in zip(views,angles):
    px,py,vis=project(a,S_LR); x0=np.floor(px).astype(int); y0=np.floor(py).astype(int)
    ok=(x0>=0)&(x0<S_LR)&(y0>=0)&(y0<S_LR)
    c=im[y0[ok],x0[ok]]
    w=np.clip(np.cos(th[vis][ok]-a),0,1)**0.5
    uu=((th[vis][ok]+np.pi)/(2*np.pi)*(TWt-1)).astype(int)
    vv=((h[vis][ok]+1)/2*(THt-1)).astype(int)
    idx=vv*TWt+uu
    for ch in range(3): acc[...,ch]+=np.bincount(idx,weights=c[:,ch]*w,minlength=THt*TWt).reshape(THt,TWt)
    wacc+=np.bincount(idx,weights=w,minlength=THt*TWt).reshape(THt,TWt)
tex=acc/np.maximum(wacc,1e-6)[...,None]
vh=np.clip(((h+1)/2*(THt-1)).astype(int),0,THt-1); uh=np.clip(((th+np.pi)/(2*np.pi)*(TWt-1)).astype(int),0,TWt-1)
re,mre=render(tex[vh,uh],0.0,S_HR)
def ps(a):
    e=((a-gt)**2*mgt[...,None]).sum()/max(mgt.sum()*3,1); return 10*np.log10(1/max(e,1e-12))
edge=np.zeros_like(mgt)
for y in range(S_HR):
    xs=np.where(mgt[y])[0]
    if len(xs)>4: edge[y,xs[:10]]=True; edge[y,xs[-10:]]=True
inter=mgt&~edge
def psr(a,msk):
    e=((a-gt)**2*msk[...,None]).sum()/max(msk.sum()*3,1); return 10*np.log10(1/max(e,1e-12))
print("真实纹理 全图: 双三次 %.2f | 平面 %.2f | 纹理空间 %.2f"%(ps(bi),ps(flat_up),ps(re)))
print("边缘带: 双三次 %.2f | 平面 %.2f | 纹理 %.2f"%(psr(bi,edge),psr(flat_up,edge),psr(re,edge)))
print("内部:   双三次 %.2f | 平面 %.2f | 纹理 %.2f"%(psr(bi,inter),psr(flat_up,inter),psr(re,inter)))
canv=Image.new('RGB',(S_HR*4+24,S_HR+36),(18,18,18)); ft=ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc',18)
for k,(im,lab) in enumerate(((Image.fromarray((bi*255).astype(np.uint8)),"双三次"),(Image.fromarray((flat_up*255).astype(np.uint8)),"平面累积"),
                             (Image.fromarray((re*255).astype(np.uint8)),"纹理空间累积"),(Image.fromarray((gt*255).astype(np.uint8)),"真实高分辨率"))):
    canv.paste(im,(k*(S_HR+8),36)); ImageDraw.Draw(canv).text((k*(S_HR+8)+4,8),lab,font=ft,fill=(255,255,255))
canv.save('/tmp/realtex_result.png'); print("saved")
