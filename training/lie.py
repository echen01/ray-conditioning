import torch
import numpy as np

# Taken from BARF (Bundle Adjusting Radiance Fields)

def so3_to_SO3(w):  # [...,3]
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    R = I + A * wx + B * wx @ wx
    return R


def SO3_to_so3(R, eps=1e-7):  # [...,3,3]
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[
        ..., None, None
    ] % np.pi  # ln(R) will explode if theta==pi
    lnR = (
        1 / (2 * taylor_A(theta) + 1e-8) * (R - R.transpose(-2, -1))
    )  # FIXME: wei-chiu finds it weird
    w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
    w = torch.stack([w0, w1, w2], dim=-1)
    return w


def se3_to_SE3(wu):  # [...,3]
    w, u = wu.split([3, 3], dim=-1)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    R = I + A * wx + B * wx @ wx
    V = I + B * wx + C * wx @ wx
    Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
    return Rt


def SE3_to_se3(Rt, eps=1e-8):  # [...,3,4]
    R, t = Rt.split([3, 1], dim=-1)
    w = SO3_to_so3(R)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta**2 + eps) * wx @ wx
    u = (invV @ t)[..., 0]
    wu = torch.cat([w, u], dim=-1)
    return wu # [..., 6]


def skew_symmetric(w):
    w0, w1, w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack(
        [
            torch.stack([O, -w2, w1], dim=-1),
            torch.stack([w2, O, -w0], dim=-1),
            torch.stack([-w1, w0, O], dim=-1),
        ],
        dim=-2,
    )
    return wx


def taylor_A(x, nth=10):
    # Taylor expansion of sin(x)/x
    ans = torch.zeros_like(x)
    denom = 1.0
    for i in range(nth + 1):
        if i > 0:
            denom *= (2 * i) * (2 * i + 1)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def taylor_B(x, nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = torch.zeros_like(x)
    denom = 1.0
    for i in range(nth + 1):
        denom *= (2 * i + 1) * (2 * i + 2)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def taylor_C(x, nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = torch.zeros_like(x)
    denom = 1.0
    for i in range(nth + 1):
        denom *= (2 * i + 2) * (2 * i + 3)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans
