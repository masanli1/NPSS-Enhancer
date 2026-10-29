import numpy as np, sys
sys.path.insert(0,'/Users/ma/Downloads/降噪/NPSS-Enhancer-repo/GUI')
import npss_image as M
from PIL import Image
HR=np.array(Image.open('/Users/ma/Downloads/降噪/图像降噪/NPSS-Best/kodim23.png').convert('RGB')).astype(np.float32)/255
def downsample(a,f=2):
    h,w=a.shape[:2]
    return a[:h//f*f,:w//f*f].reshape(h//f,f,w//f,f,3).mean((1,3))
def fshift(a,dy,dx):
    F=np.fft.fft2(a,axes=(0,1))
    fy=np.fft.fftfreq(a.shape[0])[:,None]; fx=np.fft.fftfreq(a.shape[1])[None,:]
    return np.real(np.fft.ifft2(F*np.exp(-2j*np.pi*(fy*dy+fx*dx))[...,None],axes=(0,1)))
def drizzle(frames, shifts, f=2, sign=-1.0):
    h,w,_=frames[0].shape; H,W=h*f,w*f
    num=np.zeros((H,W,3),np.float64); den=np.zeros((H,W,1),np.float64)
    yy,xx=np.mgrid[0:h,0:w].astype(np.float64)
    for img,(dy,dx) in zip(frames,shifts):
        hy=(yy+sign*dy)*f; hx=(xx+sign*dx)*f
        y0=np.floor(hy).astype(np.int32); x0=np.floor(hx).astype(np.int32)
        fy=(hy-y0); fx=(hx-x0)
        for oy in (0,1):
            for ox in (0,1):
                wy=(1-fy) if oy==0 else fy
                wx=(1-fx) if ox==0 else fx
                wgt=(wy*wx)[...,None]
                yi=np.clip(y0+oy,0,H-1); xi=np.clip(x0+ox,0,W-1)
                idx=(yi*W+xi).ravel()
                for c in range(3):
                    num[...,c]+=np.bincount(idx,weights=(img[...,c]*wgt[...,0]).ravel(),minlength=H*W).reshape(H,W)
                den[...,0]+=np.bincount(idx,weights=wgt[...,0].ravel(),minlength=H*W).reshape(H,W)
    return np.where(den>1e-8, num/np.maximum(den,1e-8), 0)
LR=downsample(HR,2); ref=HR[:LR.shape[0]*2,:LR.shape[1]*2]
def psnr(a,b):
    m=min(a.shape[0],b.shape[0]); n=min(a.shape[1],b.shape[1])
    return M.psnr(a[:m,:n],b[:m,:n])
base=np.array(Image.fromarray(M.to_uint8(LR)).resize((LR.shape[1]*2,LR.shape[0]*2),Image.BICUBIC)).astype(np.float32)/255
print("单帧双三次上采样 %.2f dB"%psnr(base,ref))
for N in (4,8,16,32,64):
    rng=np.random.default_rng(0)
    shifts=[rng.uniform(0,1,2) for _ in range(N)]
    # 正确模拟：先把 HR 平移亚像素，再下采样（相机在不同相位采样同一场景）
    frames=[np.clip(downsample(np.clip(fshift(HR,dy*2,dx*2),0,1),2),0,1) for dy,dx in shifts]
    # 对齐回参考帧（统一减去各自位移）
    for sg in (-1.0, 1.0):
        hr=drizzle(frames, shifts, 2, sg)
        print("drizzle %2d 帧 sign%+.0f %.2f dB"%(N,sg,psnr(hr,ref)))
