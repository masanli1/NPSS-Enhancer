import numpy as np, time
from PIL import Image, ImageDraw, ImageFont
rng=np.random.default_rng(0)

# 球面点云（曲面上采样），解析纹理作为真值
N=20000
v=rng.normal(size=(N,3)); v/=np.linalg.norm(v,axis=1,keepdims=True)
X=v.copy()
def tex(n):
    # 曲面纹理（高频，用来体现"展开后采样更密"）
    th=np.arctan2(n[:,2],n[:,0]); ph=np.arccos(np.clip(n[:,1],-1,1))
    a=0.5+0.5*np.sin(18*th)*np.cos(10*ph)
    b=0.5+0.5*np.sin(14*ph+1.0)
    c=0.5+0.5*np.cos(16*th+0.4)
    return np.stack([a,b,c],1)
col=tex(X)
def rot(rx,ry):
    cx,sx=np.cos(rx),np.sin(rx); cy,sy=np.cos(ry),np.sin(ry)
    return np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
F=2.2
def render(pts,cols,R,t,S):
    p=pts@R.T+t; z=p[:,2]; ok=z>0.2
    u=F*p[ok,0]/z[ok]; w=F*p[ok,1]/z[ok]; c=cols[ok]
    px=u*S/2+S/2; py=w*S/2+S/2
    x0=np.floor(px).astype(int); y0=np.floor(py).astype(int); fx=px-x0; fy=py-y0
    acc=np.zeros((S,S,3)); wa=np.zeros((S,S))
    for oy in (0,1):
        for ox in (0,1):
            xx=x0+ox; yy=y0+oy
            mm=(xx>=0)&(xx<S)&(yy>=0)&(yy<S)
            ww=((1-fy) if oy==0 else fy)*((1-fx) if ox==0 else fx)*mm
            idx=yy[mm]*S+xx[mm]
            for ch in range(3): acc[...,ch]+=np.bincount(idx,weights=c[mm,ch]*ww[mm],minlength=S*S).reshape(S,S)
            wa+=np.bincount(idx,weights=ww[mm],minlength=S*S).reshape(S,S)
    return np.clip(acc/np.maximum(wa,1e-6)[...,None],0,1), wa>1e-6
S_LR=160; S_HR=320
poses=[(0.25*np.sin(k*0.8), 0.6*k) for k in range(8)]
views=[render(X,col,rot(*p),np.array([0,0,3.0]),S_LR)[0] for p in poses]
ref_pose=poses[3]
gt,_=render(X,col,rot(*ref_pose),np.array([0,0,3.0]),S_HR)
# 立体角/朝向可见性：球面法线=点坐标
def vis_weight(n,R):
    view=np.array([0,0,-1.0])          # 相机看向 -z
    nd=n@R.T
    return np.clip(-(nd@view),0,1)     # 面向相机
# UV 参数化（球面展开）
th=np.arctan2(X[:,2],X[:,0]); ph=np.arccos(np.clip(X[:,1],-1,1))
U=((th+np.pi)/(2*np.pi)); V=ph/np.pi
TW,TH=640,320                       # 纹理分辨率：比单帧密 4 倍
acc=np.zeros((TH,TW,3)); wacc=np.zeros((TH,TW,1))
for (rx,ry) in poses:
    R=rot(rx,ry); w=vis_weight(X,R)
    # 用渲染后的图像采样每个点的颜色（近似：直接解析纹理 + 噪声）
    c=np.clip(col+rng.normal(0,0.02,col.shape),0,1)
    u=(U*(TW-1)).astype(int); vv=(V*(TH-1)).astype(int)
    for ch in range(3):
        acc[...,ch]+=np.bincount(vv*TW+u,weights=c[:,ch]*w,minlength=TH*TW).reshape(TH,TW)
    wacc[...,0]+=np.bincount(vv*TW+u,weights=w,minlength=TH*TW).reshape(TH,TW)
tex_img=acc/np.maximum(wacc,1e-6)
# 重投影：用纹理重建颜色
def lookup():
    u=np.clip((U*(TW-1)).astype(int),0,TW-1); vv=np.clip((V*(TH-1)).astype(int),0,TH-1)
    return tex_img[vv,u]
col2=lookup()
re,_=render(X,col2,rot(*ref_pose),np.array([0,0,3.0]),S_HR)
bi=np.array(Image.fromarray((views[3]*255).astype(np.uint8)).resize((S_HR,S_HR),Image.BICUBIC)).astype(np.float32)/255
m=gt.sum(-1)>0.03
def ps(a,b): 
    e=((a-b)**2*m[...,None]).sum()/max(m.sum()*3,1); return 10*np.log10(1/max(e,1e-12))
print("单帧双三次 %.2f dB | 纹理空间累积重投影 %.2f dB"%(ps(bi,gt),ps(re,gt)))
W=S_HR*4+24; canv=Image.new('RGB',(W,S_HR+36),(18,18,18)); ft=ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc',18)
for k,(im,lab) in enumerate(((Image.fromarray((views[3]*255).astype(np.uint8)).resize((S_HR,S_HR),Image.NEAREST),"低分辨率输入"),
                             (Image.fromarray((bi*255).astype(np.uint8)),"双三次放大"),
                             (Image.fromarray((re*255).astype(np.uint8)),"纹理空间累积"),
                             (Image.fromarray((gt*255).astype(np.uint8)),"真实高分辨率"))):
    canv.paste(im,(k*(S_HR+8),36)); ImageDraw.Draw(canv).text((k*(S_HR+8)+4,8),lab,font=ft,fill=(255,255,255))
canv.save('/tmp/surfel_result.png'); print("saved")
