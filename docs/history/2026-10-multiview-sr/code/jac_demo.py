import numpy as np
exec(open('/tmp/rim_demo.py').read().split("angles=[")[0])
S=192
a0=-0.2
f0,m0=render(col,a0,S,noise=0.02)
f1,m1=render(col,a0+0.09,S,noise=0.02)
# 块匹配光流
B=8; R=5
H=W=S
gy,gx=np.mgrid[0:B,0:B]
u=np.zeros((H//B,W//B)); v=np.zeros((H//B,W//B))
g0=f0.mean(-1); g1=f1.mean(-1)
for by in range(H//B):
    for bx in range(W//B):
        y0,x0=by*B,bx*B
        ref=g0[y0:y0+B,x0:x0+B]
        best=(1e9,0,0)
        for dy in range(-R,R+1):
            for dx in range(-R,R+1):
                yy=y0+dy; xx=x0+dx
                if yy<0 or xx<0 or yy+B>H or xx+B>W: continue
                cand=g1[yy:yy+B,xx:xx+B]
                e=((cand-ref)**2).mean()
                if e<best[0]: best=(e,dy,dx)
        v[by,bx],u[by,bx]=best[1],best[2]
# 雅可比（块中心差分）
du_dx=np.gradient(u,axis=1); dv_dy=np.gradient(v,axis=0)
du_dy=np.gradient(u,axis=0); dv_dx=np.gradient(v,axis=1)
det=(1+du_dx)*(1+dv_dy)-du_dy*dv_dx
det=np.kron(det,np.ones((B,B)))[:H,:W]
inside=np.kron(m0[::B,::B].astype(float),np.ones((B,B)))[:H,:W]>0.5
print("面积压缩区域（det<0.9）占主体比例: %.1f%%"%(100*((det<0.9)&inside).sum()/max(inside.sum(),1)))
# 真实的前向压缩率：表面点朝向 cos(θ+a0)
th_lim=np.abs(np.arctan2(np.sin(th),np.cos(th)))  # 仅用于对照
xs=np.arange(W)
# 解析：圆柱在屏上的横向压缩 = cos(视角)
print("检测到的压缩区中心位置: 左 %.0f%%  右 %.0f%%（屏宽比例）"%(
    100*(np.where(((det<0.9)&inside).any(0))[0].min()/W),
    100*(np.where(((det<0.9)&inside).any(0))[0].max()/W)))
