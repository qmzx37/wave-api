# wave_equation_sim.py
# --------------------------------------------
# 2차(감쇠 진동) 파동/상태 방정식 시뮬레이션:
#   x'' + 2ζω x' + ω^2 x = ω^2 u(t)
#
# - u(t): "입력(감정축 목표값/자극)" (0~1)
# - x(t): "psi(상태)" (0~1)
# - ω: 빠르기(긴장 T가 높을수록 커짐)
# - ζ: 감쇠(안정 R이 높을수록 커짐)
#
# 실행:
#   PS> cd C:\llm\train\wave
#   PS> python .\wave_equation_sim.py
# --------------------------------------------

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def omega_from_T(T: float) -> float:
    # AB 테스트에서 FAST(0.9)≈12.1, SLOW(0.1)≈2.5처럼 나오게 한 스케일
    T = clamp01(T)
    return 1.3 + 12.0 * T  # rad/s


def zeta_from_R(R: float) -> float:
    # R=0.9 -> ~0.88, R=0.1 -> ~0.05 근처
    R = clamp01(R)
    return clamp01(0.02 + 1.02 * (R ** 1.6))


def simulate(T: float, R: float, dt: float = 0.01, seconds: float = 12.0):
    """
    입력 u(t)를 몇 구간으로 바꿔가며 psi(x)가 어떻게 따라오는지 보기
    """
    w = omega_from_T(T)
    z = zeta_from_R(R)

    n = int(seconds / dt)
    t = np.arange(n) * dt

    # 입력 u(t): (원하면 네 로그/축을 넣어도 됨)
    u = np.zeros(n, dtype=np.float32)
    u[(t >= 1.0) & (t < 4.0)] = 0.80  # 자극 상승
    u[(t >= 4.0) & (t < 7.0)] = 0.20  # 자극 감소
    u[(t >= 7.0)] = 0.60  # 다시 중간

    x = np.zeros(n, dtype=np.float32)   # psi
    v = np.zeros(n, dtype=np.float32)   # psi'

    # 초기값
    x[0] = 0.05
    v[0] = 0.00

    # semi-implicit Euler (안정적)
    for i in range(1, n):
        # x'' = ω^2(u - x) - 2ζω v
        a = (w * w) * (float(u[i-1]) - float(x[i-1])) - 2.0 * z * w * float(v[i-1])
        v_i = float(v[i-1]) + a * dt
        x_i = float(x[i-1]) + v_i * dt

        # 클램프 + 속도 제한(너무 튀면 대화가 이상해짐)
        v_i = max(-3.0, min(3.0, v_i))
        x_i = clamp01(x_i)

        v[i] = v_i
        x[i] = x_i

    return t, u, x, v, w, z


def main():
    cases = [
        ("FAST (T=0.90, R=0.10)", 0.90, 0.10),
        ("SLOW (T=0.10, R=0.90)", 0.10, 0.90),
        ("MID  (T=0.45, R=0.45)", 0.45, 0.45),
    ]

    plt.figure()
    for name, T, R in cases:
        t, u, x, v, w, z = simulate(T, R)
        plt.plot(t, x, label=f"{name} | ω={w:.2f}, ζ={z:.2f}")

    plt.plot(t, u, linestyle="--", label="u(t) input")
    plt.xlabel("t (sec)")
    plt.ylabel("value (0~1)")
    plt.title("2nd-order psi dynamics: x'' + 2ζω x' + ω^2 x = ω^2 u(t)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
