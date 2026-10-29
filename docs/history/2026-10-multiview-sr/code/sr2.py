import numpy as np, sys
sys.path.insert(0,'/Users/ma/Downloads/降噪/图像降噪/NPSS-Best')
import npss_image as M
from PIL import Image
HR=np.array(Image.open('/Users/ma/Downloads/降噪/图像降噪/NPSS-Best/kodim23.png').convert('RGB')).astype(np.float32)/255
def downsample(a,f=2):
    h,w=a.shape[:2]
    return a[:h//f*f,:w//f*f].reshape(h//f,f,w//f,f,3).mean((1,3))
def fshift(a,dy,dx):
    F=np.fft.fft2(a,axes=(0,1)); H,W=a.shape[:2]
    fy=np.fft.fftfreq(H)[:,None]; fx=np.fft.fftfreq(W)[None,:]
    return np.real(np.fft.ifft2(F*np.exp(-2j*np.pi*(fy*dy+fx*dx))[...,None],axes=(0,1)))
def fup(a,f=2):
    H,W,_=a.shape
    F=np.fft.fft2(a,axes=(0,1))
    F=np.fft.fftshift(F,axes=(0,1))
    pad=np.zeros((H*f,W*f,3),complex)
    y0=H*f//2-H//2; x0=W*f//2-W//2
    pad[y0:y0+H,x0:x0+W]=F
    pad=np.fft.ifftshift(pad,axes=(0,1))
    return np.real(np.fft.ifft2(pad,axes=(0,1)))*(f*f)
LR=downsample(HR,2); ref=HR[:LR.shape[0]*2,:LR.shape[1]*2]
def psnr(a,b):
    m=min(a.shape[0],b.shape[0]); n=min(a.shape[1],b.shape[1])
    return M.psnr(np.clip(a[:m,:n],0,1),b[:m,:n])
up=fup(LR,2)
print("fup vs HR %.2f"%psnr(up,ref))
print("fup vs bicubic %.2f"%psnr(up,np.array(Image.fromarray(M.to_uint8(LR)).resize((LR.shape[1]*2,LR.shape[0]*2),Image.BICUBIC)).astype(np.float32)/255))
for N in (4,8,16,32,64):
    rng=np.random.default_rng(0); shifts=[rng.uniform(0,1,2) for _ in range(N)]
    frames=[np.clip(downsample(np.clip(fshift(HR,dy*2,dx*2),0,1),2),0,1) for dy,dx in shifts]
    acc=np.zeros_like(up)
    for fr,(dy,dx) in zip(frames,shifts):
        acc+=fup(fshift(fr,-dy,-dx),2)
    acc/=N
    print("shift-add %2d 帧 %.2f"%(N,psnr(acc,ref)))
