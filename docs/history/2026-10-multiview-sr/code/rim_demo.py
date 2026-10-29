import numpy as np
from PIL import Image, ImageDraw, ImageFont
rng=np.random.default_rng(0)

# ---------- 圆柱（半径1，高2），表面高频纹理 ----------
N=40000
th=rng.uniform(-np.pi,np.pi,N); h=rng.uniform(-1,1,N)
def cyl_tex(th,h):
    a=0.5+0.5*np.sin(16*th)*np.cos(6*h)
    b=0.5+0.5*np.sin(11*h+0.8)
    c=0.5+0.5*np.cos(18*th+0.3)
    return np.stack([a,b,c],1)
col=cyl_tex(th,h)
F=2.2; S_LR=192; S_HR=384
CAM=np.array([0,0,3.2])

def project(Rz, S):
    x=np.sin(th)*np.cos(Rz)+np.cos(th)*np.sin(Rz)   # 绕 y 轴？这里用绕竖轴旋转: x=sin(θ+rot)
    z=-(np.sin(th)*np.sin(Rz))+np.cos(th)*np.cos(Rz)
    X=np.stack([x,h,z],1)                            # 圆柱侧表面
    X=X+CAM
    vis=z>0.05
    u=F*X[vis,0]/X[vis,2]; v=F*X[vis,1]/X[vis,2]
    px=u*S/2+S/2; py=v*S/2+S/2
    return px,py,vis

def render(colors, Rz, S, noise=0.0):
    px,py,vis=project(Rz,S)
    c=colors[vis]
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

# ---------- A. 单帧双三次 ----------
bi=np.array(Image.fromarray((views[0][0]*255).astype(np.uint8)).resize((S_HR,S_HR),Image.BICUBIC)).astype(np.float32)/255

# ---------- B. 平面累积：把每帧当作平面贴片，用 2D 平移对齐后平均 ----------
flat_acc=np.zeros((S_LR,S_LR,3)); n=0
for img,m in views:
    # 用轮廓中心对齐（近似）
    ys,xs=np.where(m)
    if len(xs)==0: continue
    dx=int(round(xs.mean()-S_LR/2)); dy=int(round(ys.mean()-S_LR/2))
    flat_acc+=np.roll(np.roll(img,-dy,0),-dx,1); n+=1
flat=flat_acc/n
flat_up=np.array(Image.fromarray((flat*255).astype(np.uint8)).resize((S_HR,S_HR),Image.BICUBIC)).astype(np.float32)/255

# ---------- C. 边缘弧度建模：纹理空间 (θ, h)，只对可见半周累积 ----------
TW,TH=768,384                      # 纹理比单帧密 4 倍
acc=np.zeros((TH,TW,3)); wacc=np.zeros((TH,TW))
for (img,m),a in zip(views,angles):
    px,py,vis=project(a,S_LR)      # 每个表面点投影到该帧
    x0=np.floor(px).astype(int); y0=np.floor(py).astype(int)
    ok=(x0>=0)&(x0<S_LR)&(y0>=0)&(y0<S_LR)
    sample=m[y0[ok],x0[ok]]
    cols=img[y0[ok],x0[ok]]
    # 权重：越靠边缘（朝向越斜）权重越低
    w=np.clip(np.cos(th[vis][ok]-a),0,1)**0.5
    u=((th[vis][ok]+np.pi)/(2*np.pi)*(TW-1)).astype(int)
    v=((h[vis][ok]+1)/2*(TH-1)).astype(int)
    idx=v*TW+u
    for ch in range(3):
        acc[...,ch]+=np.bincount(idx,weights=cols[:,ch]*w,minlength=TH*TW).reshape(TH,TW)
    wacc+=np.bincount(idx,weights=w,minlength=TH*TW).reshape(TH,TW)
tex_img=acc/np.maximum(wacc,1e-6)[...,None]
vh=np.clip(((h+1)/2*(TH-1)).astype(int),0,TH-1); uh=np.clip(((th+np.pi)/(2*np.pi)*(TW-1)).astype(int),0,TW-1)
col_hat=tex_img[vh,uh]
re,mre=render(col_hat,0.0,S_HR)

mgt2=mgt
def ps(a):
    e=((a-gt)**2*mgt2[...,None]).sum()/max(mgt2.sum()*3,1)
    return 10*np.log10(1/max(e,1e-12))
print("单帧双三次 %.2f | 平面累积 %.2f | 弧度纹理累积 %.2f"%(ps(bi),ps(flat_up),ps(re)))
# 边缘带 vs 内部
ys,xs=np.where(mgt2); cx=W2=None
xr=xs.max()-xs[0] if len(xs) else 0
W=384
edge=np.zeros_like(mgt2)
for y in range(W):
    xr_=np.where(mgt2[y])[0]
    if len(xr_)>4:
        edge[y,xr_[:6]]=True; edge[y,xr_[-6:]]=True
def ps_region(a,msk):
    e=((a-gt)**2*msk[...,None]).sum()/max(msk.sum()*3,1); return 10*np.log10(1/max(e,1e-12))
print("边缘带: 双三次 %.2f | 平面累积 %.2f | 弧度 %.2f"%(ps_region(bi,edge),ps_region(flat_up,edge),ps_region(re,edge)))
interior=mgt2&~edge
print("内部:   双三次 %.2f | 平面累积 %.2f | 弧度 %.2f"%(ps_region(bi,interior),ps_region(flat_up,interior),ps_region(re,interior)))
canv=Image.new('RGB',(S_HR*4+24,S_HR+36),(18,18,18)); ft=ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc',18)
for k,(im,lab) in enumerate(((Image.fromarray((views[0][0]*255).astype(np.uint8)).resize((S_HR,S_HR),Image.NEAREST),"低分辨率输入"),
                             (Image.fromarray((bi*255).astype(np.uint8)),"双三次"),
                             (Image.fromarray((flat_up*255).astype(np.uint8)),"平面累积"),
                             (Image.fromarray((re*255).astype(np.uint8)),"弧度纹理累积"))):
    canv.paste(im,(k*(S_HR+8),36)); ImageDraw.Draw(canv).text((k*(S_HR+8)+4,8),lab,font=ft,fill=(255,255,255))
canv.save('/tmp/rim_result.png'); print("saved")
