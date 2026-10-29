import numpy as np, time
from PIL import Image
from PIL import ImageDraw, ImageFont
rng=np.random.default_rng(0)
N=4000
v=rng.normal(size=(N,3)); v/=np.linalg.norm(v,axis=1,keepdims=True)
X=v.copy()
col=np.stack([0.5+0.5*np.sin(4*v[:,0])*np.cos(4*v[:,1]),
              0.5+0.5*np.sin(5*v[:,1]+1.3),
              0.5+0.5*np.cos(6*v[:,2]+0.7)],1).clip(0,1)
def rot(rx,ry):
    cx,sx=np.cos(rx),np.sin(rx); cy,sy=np.cos(ry),np.sin(ry)
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]); Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    return Ry@Rx
F=2.2
def render(pts,cols,R,t,S):
    p=pts@R.T+t; z=p[:,2]
    ok=z>0.2
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
            for ch in range(3):
                acc[...,ch]+=np.bincount(idx,weights=c[mm,ch]*ww[mm],minlength=S*S).reshape(S,S)
            wa+=np.bincount(idx,weights=ww[mm],minlength=S*S).reshape(S,S)
    return np.clip(acc/np.maximum(wa,1e-6)[...,None],0,1), wa>1e-6
S_LR=160; S_HR=320
poses=[(0.3*np.sin(k*0.7), 0.5*k) for k in range(8)]
obs=[];msk=[]
for rx,ry in poses:
    im,m=render(X,col,rot(rx,ry),np.array([0,0,3.0]),S_LR)
    obs.append(np.clip(im+rng.normal(0,0.01,im.shape),0,1)); msk.append(m)
gt,mgt=render(X,col,rot(poses[3][0],poses[3][1]),np.array([0,0,3.0]),S_HR)
def err(img,m,ref):
    return float((((img-ref)**2)*m[...,None]).sum()/max(m.sum()*3,1))
def bicubic(img,S):
    return np.array(Image.fromarray((img*255).astype(np.uint8)).resize((S,S),Image.BICUBIC)).astype(np.float32)/255
# 猜测模型：形状略有偏差（沿 3 个轴不同缩放）+ 噪声
Xg=X*0.85              # 猜测模型：形状正确、尺度偏小 15%
def fit(i):
    best=(1e9,None)
    # 粗搜
    for rx in np.linspace(0,2*np.pi,24,endpoint=False):
        for ry in np.linspace(0,2*np.pi,24,endpoint=False):
            for sc in (0.8,0.9,1.0,1.1,1.2):
                im,m=render(Xg*sc,col,rot(rx,ry),np.array([0,0,3.0]),S_LR)
                e=err(im,m,obs[i])
                if e<best[0]: best=(e,[rx,ry,sc,0.0,0.0])
    th=best[1]
    # 精修
    step=np.array([0.12,0.12,0.06,0.05,0.05])
    for it in range(60):
        improved=False
        for j in range(5):
            for s in (1,-1):
                t2=th.copy(); t2[j]+=s*step[j]
                im,m=render(Xg*t2[2],col,rot(t2[0],t2[1]),np.array([t2[3],t2[4],3.0]),S_LR)
                e=err(im,m,obs[i])
                if e<best[0]: best=(e,t2); th=t2; improved=True
        if not improved: step*=0.6
    return best[1],best[0]
t0=time.time(); fits=[fit(i) for i in range(len(obs))]
print("拟合 %.1fs  误差 首帧 %.4f 末帧 %.4f"%(time.time()-t0,fits[0][1],fits[-1][1]))
th,sc=fits[3][0],fits[3][0][2]
re,_=render(Xg*sc,col,rot(th[0],th[1]),np.array([th[3],th[4],3.0]),S_HR)
bi=bicubic(obs[3],S_HR)
def ps(a,b,m): return 10*np.log10(1.0/max(err(a,m,b),1e-12))
print("第4帧 双三次 %.2f dB | 模型重投影 %.2f dB"%(ps(bi,gt,mgt),ps(re,gt,mgt)))
# 拼图
def to_img(a,S):
    im=Image.fromarray((a*255).astype(np.uint8))
    return im
a=Image.fromarray((obs[3]*255).astype(np.uint8)).resize((S_HR,S_HR),Image.NEAREST)
b=to_img(bi,S_HR); c=to_img(re,S_HR); d=to_img(gt,S_HR)
W=S_HR*4+3*8; canv=Image.new('RGB',(W,S_HR+36),(18,18,18))
ft=ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc',18)
for k,(im,lab) in enumerate(((a,"低分辨率输入"),(b,"双三次放大"),(c,"三维模型重投影"),(d,"真实高分辨率"))):
    canv.paste(im,(k*(S_HR+8),36)); ImageDraw.Draw(canv).text((k*(S_HR+8)+4,8),lab,font=ft,fill=(255,255,255))
canv.save('/tmp/abs_result.png'); print("saved")
