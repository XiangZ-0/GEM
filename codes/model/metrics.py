from sewar.full_ref import psnr, ssim, mse

def compare_psnr(x_true, x_pred):
    return psnr(x_true, x_pred)

def compare_ssim(x_true, x_pred):
    return ssim(x_true, x_pred)[0]

