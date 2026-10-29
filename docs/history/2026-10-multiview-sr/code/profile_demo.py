import numpy as np
from PIL import Image, ImageDraw, ImageFont
rng=np.random.default_rng(0)
# 复用圆柱渲染
exec(open('/tmp/rim_demo.py').read().split("angles=[")[0])
S_LR=192; S_HR=384
angles=[-0.5+0.09*k for k in range(12)]
views=[render(col,a,S_LR,noise=0.03) for a in angles]
gt,mgt=render(col,0.0,S_HR)

def edges(m):
    """每行左右边界 x（亚像素用亮度梯度细化）"""
    L=np.full(m.shape[0],np.nan); R=np.full(m.shape[0],np.nan)
    for y in range(m.shape[0]):
        xs=np.where(m[y])[0]
        if len(xs)>2: L[y]=xs[0]; R[y]=xs[-1]
    return L,R

def profiles(img,m,cy=0.5,half=8):
    """在中间若干行上，取左右边缘两侧的法向剖面"""
    outL=[];outR=[]
    L,R=edges(m)
    ys=[y for y in range(int(m.shape[0]*0.35),int(m.shape[0]*0.65)) if not np.isnan(L[y])]
    for y in ys:
        xl=int(round(L[y])); xr=int(round(R[y]))
        outL.append(img[y,xl-half:xl+half].mean(-1))
        outR.append(img[y,xr-half:xr+half].mean(-1))
    return np.array(outL),np.array(outR)

def align_1d(a,b):
    """互相关 + 抛物线亚像素，返回把 b 对齐到 a 的位移"""
    A=a-a.mean(); B=b-b.mean()
    c=np.correlate(A,B,'full')
    i=int(np.argmax(c))
    d=0.0
    if 0<i<len(c)-1:
        y0,y1,y2=c[i-1],c[i],c[i+1]
        d=0.5*(y0-y2)/(y0-2*y1+y2+1e-12)
    return (i-(len(a)-1))+d

# 参考帧（第 6 帧）
half=8
ref_img,ref_m=views[6]
refP,_=profiles(ref_img,ref_m,half=half)
# 各帧剖面 → 对齐 → 4x 重采样累积
F=4; up=64
acc=np.zeros((refP.shape[0],up)); cnt=np.zeros((refP.shape[0],up))
base=np.zeros((refP.shape[0],up))
for img,m in views:
    P,_=profiles(img,m)
    n=min(len(P),len(refP))
    for row in range(n):
        p=P[row]
        d=align_1d(refP[row],p)                    # 亚像素位移
        x=np.arange(up)/F - half + 0.5/F           # 4x 网格（LR 像素坐标）
        sampled=np.interp(x+d, np.arange(len(p)), p, left=np.nan,right=np.nan)
        ok=~np.isnan(sampled)
        acc[row,ok]+=sampled[ok]; cnt[row,ok]+=1
rec=np.where(cnt>0,acc/np.maximum(cnt,1),0)
# 单帧上采样基线
base=np.array([np.interp(np.arange(up)/F-half+0.5/F, np.arange(len(r)), r) for r in refP])
# 真值：HR 渲染在同样行的剖面
gtP,_=profiles(gt,mgt,half=half*2)
gtP=np.array([np.interp(np.arange(up)/F-half+0.5/F, np.arange(len(r))/2, r) for r in gtP])
n=min(len(rec),len(gtP))
def ps(a,b):
    e=((a[:n]-b[:n])**2).mean(); return 10*np.log10(1/max(e,1e-12))
print("边缘剖面 4x: 单帧插值 %.2f dB | 多帧剖面累积 %.2f dB"%(ps(base,gtP),ps(rec,gtP)))
# 可视化
def strip(a,w=8):
    a=np.clip(a,0,1)
    return (np.repeat(np.repeat(a[:,None,:],w,1),1,0)*255).astype(np.uint8)
canv=Image.new('RGB',(up*4+30,refP.shape[0]+36),(18,18,18))
ft=ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc',16)
for k,(im,lab) in enumerate(((strip(base),"单帧插值4x"),(strip(rec),"剖面累积4x"),(strip(gtP),"真值4x"))):
    canv.paste(Image.fromarray(im),(k*(up+10),30)); ImageDraw.Draw(canv).text((k*(up+10),6),lab,font=ft,fill=(255,255,255))
canv.save('/tmp/profile_result.png'); print("saved")
