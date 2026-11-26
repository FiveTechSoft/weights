"""Exact continuous gradient flow of a balanced scalar two-layer linear net.

Loss = (a*b-c)^2/2 for unit input; a(0)=b(0)>0, c>0.
This is NOT exact finite-step SGD, AdamW, or general nonlinear training.
"""
import math


def balanced_flow(a, b, target, time):
    if not all(math.isfinite(v) for v in (a, b, target, time)):
        raise ValueError("nonfinite input")
    if a != b or a <= 0 or target <= 0 or time < 0:
        raise ValueError("requires positive balanced weights, positive target, nonnegative time")
    if time == 0:
        return a, b
    # log-domain evaluation avoids overflow in target / a**2.
    decay = 2 * target * time
    if math.isinf(decay):
        w = math.sqrt(target)
        return w, w
    # p(t) = c / [1-exp(-2ct)+(c/p0)*exp(-2ct)]
    log_u = math.log(-math.expm1(-decay)) if decay > 0 else -math.inf
    log_v = math.log(target) - 2 * math.log(a) - decay
    m = max(log_u, log_v)
    log_den = m + math.log(math.exp(log_u-m) + math.exp(log_v-m))
    w = math.exp((math.log(target)-log_den)/2)
    return w, w


def self_test():
    import json
    from scipy.integrate import solve_ivp
    results = []
    for initial, target in ((.1, 2.), (.5, 1.3), (1.5, .7)):
        t = 3.
        a, b = balanced_flow(initial, initial, target, t)
        ref = solve_ivp(lambda _, z: [(target-z[0]*z[1])*z[1],
                                    (target-z[0]*z[1])*z[0]],
                        (0,t), [initial,initial], rtol=1e-11, atol=1e-13)
        assert ref.success
        error = abs(a*b - float(ref.y[0,-1]*ref.y[1,-1]))
        assert error < 1e-9
        assert balanced_flow(initial, initial, target, 0) == (initial, initial)
        results.append(dict(initial_product=initial**2,target=target,
                            product=a*b,error=error))
    print(json.dumps(results))


if __name__ == "__main__":
    self_test()
