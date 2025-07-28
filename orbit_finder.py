import time, jax, jax.numpy as jnp
import numpy as np
import argparse

# ---------------- 1. boundary parametrisation ----------------
def curve(t, p, eps_pow=1e-14):
    ang = 2*jnp.pi*t
    q   = 2.0/p
    c, s = jnp.cos(ang), jnp.sin(ang)
    pow_ = lambda x: jnp.sign(x)*(jnp.abs(x)+eps_pow)**q
    return jnp.stack([pow_(c), pow_(s)], -1)

vcurve = jax.vmap(curve, in_axes=(0, None, None))

# ---------------- 2. length functional ----------------------
def poly_length(pts):
    d = pts - jnp.roll(pts, -1, 0)
    return jnp.sum(jnp.linalg.norm(d, axis=1))

def L(theta, p):
    return poly_length(vcurve(theta, p, 1e-14))

grad_L = jax.grad(L, 0)
hess_L = jax.hessian(L, 0)
import jax, jax.numpy as jnp
import numpy as np
import pandas as pd

import numpy as np

def mymod(x):
    """Return x modulo 1 ∈ [0,1).  Works for scalars or arrays."""
    x = np.asarray(x, dtype=float)          # guarantees ndarray
    return x - np.floor(x)

def canonical_theta(theta, reflect=True):
    theta = mymod(theta)                       # wrap once, always works

    # put smallest θ first (keeps cyclic order)
    k0 = int(np.argmin(theta))
    theta = np.roll(theta, -k0)

    if reflect and theta.size > 1:
        refl = mymod(-theta[::-1])             # reverse + reflect + wrap
        k1   = int(np.argmin(refl))            # shift its smallest first, too
        refl = np.roll(refl, -k1)
        if tuple(refl) < tuple(theta):
            theta = refl

    return tuple(theta)
# -------------- 1. Newton *single* step ----------------
@jax.jit
def newton_step(theta, p):
    g = grad_L(theta, p)
    H = hess_L(theta, p)
    d = jnp.linalg.solve(H, -g)        # N×N dense, N≤6
    return theta + d, g, H             # return g,H for last step only

# -------------- 2. k fixed Newton steps ----------------
from functools import partial
import jax, jax.numpy as jnp

@partial(jax.jit, static_argnums=2)          # ‹── tell JAX: k is static
def newton_k(theta0, p, k):
    def body(carry, _):
        theta, *_ = carry
        theta, g, H = newton_step(theta, p)
        return (theta, g, H), None
    (theta_fin, g_fin, H_fin), _ = jax.lax.scan(body,
                                               (theta0,
                                                jnp.zeros_like(theta0),
                                                jnp.zeros((theta0.size,)*2)),
                                               None,
                                               length=k)   # k is now concrete
    return theta_fin, g_fin, H_fin
# -------------- 3. vmap over seeds ---------------------
def batch_solve(seeds, p, k):
    theta_fin, g_fin, H_fin = jax.vmap(newton_k, in_axes=(0, None, None))(seeds, p, k)
    return np.array(theta_fin), np.array(g_fin), np.array(H_fin)

# -------------- 4. post-process ------------------------
def inertia_numpy(H, tol=1e-8):
    eig = np.linalg.eigvalsh(H)
    pos = int((eig >  tol).sum()); neg = int((eig < -tol).sum())
    zero= H.shape[0]-pos-neg
    return (pos,neg,zero), eig

def alpha_numpy(g, H):
    Hinv = np.linalg.inv(H)
    beta = np.linalg.norm(Hinv @ g)
    gamma= 0.5*np.linalg.norm(Hinv)*np.linalg.norm(H)
    return beta*gamma, beta, gamma

def rotation_number(theta):
    """Compute rotation number of an orbit by analyzing the permutation.
    
    For N points in parameter order, if we connect them sequentially,
    the rotation number k/N tells us the winding:
    - k=1: convex N-gon
    - k>1, gcd(k,N)=1: star polygon
    """
    theta = np.array(theta)
    n = len(theta)
    
    # Get the permutation: sorted_indices[i] tells us which original point is at position i
    sorted_indices = np.argsort(theta)
    
    # Inverse permutation: position[i] tells us where original point i ended up
    position = np.empty(n, dtype=int)
    position[sorted_indices] = np.arange(n)
    
    # Follow the orbit and count total rotation
    total_rotation = 0
    current_pos = position[0]
    
    for i in range(n):
        next_i = (i + 1) % n
        next_pos = position[next_i]
        
        # How far did we rotate from current_pos to next_pos?
        if next_pos >= current_pos:
            step = next_pos - current_pos
        else:
            step = n + next_pos - current_pos
            
        total_rotation += step
        current_pos = next_pos
    
    # The rotation number is how many times we went around
    k = total_rotation // n
    
    from math import gcd
    g = gcd(k, n)
    return k // g, n // g

# -------------- 5. main --------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute certified periodic billiard orbits')
    parser.add_argument('p', type=float, help='Shape parameter p')
    parser.add_argument('N', type=int, help='Number of bounce points')
    parser.add_argument('N_SEEDS', type=int, help='Number of random seeds to try')
    parser.add_argument('--N_STEPS', type=int, default=30, help='Number of Newton steps (default: 30)')
    
    args = parser.parse_args()
    p = args.p
    N = args.N
    N_SEEDS = args.N_SEEDS
    N_STEPS = args.N_STEPS
    
    # Validate p
    if p == 2.0:
        print("Error: p=2 corresponds to a circle, which is a degenerate case.")
        print("The variational problem is ill-posed for circular billiards.")
        print("Please use p != 2.")
        import sys
        sys.exit(1)

    rng   = np.random.default_rng(0)
    seeds = rng.random((N_SEEDS, N))     # uniform in [0,1)

    theta_fin, g_fin, H_fin = batch_solve(seeds, p, N_STEPS)
    
    orbits = {}
    for θ, g, H in zip(theta_fin, g_fin, H_fin):
        α, β, γ = alpha_numpy(g, H)
        if α < 0.15767:                 # certified
            key = canonical_theta(θ)    # robust canonicaliser
            if key not in orbits:
                sig, _ = inertia_numpy(H)
                perim  = float(L(jnp.array(key), p))
                rot_num, rot_den = rotation_number(key)
                orbits[key] = (*sig, perim, α, rot_num, rot_den)

    # Sort the orbits by their canonical theta keys
    sorted_items = sorted(orbits.items(), key=lambda x: x[0])
    
    # Coalesce nearby orbits
    coalesced = []
    if sorted_items:
        coalesced.append(sorted_items[0])
        
        for i in range(1, len(sorted_items)):
            curr_theta, curr_data = sorted_items[i]
            prev_theta, prev_data = coalesced[-1]
            
            # Calculate distance between thetas
            curr_arr = np.array(curr_theta)
            prev_arr = np.array(prev_theta)
            dist = np.linalg.norm(curr_arr - prev_arr)
            
            # Sum of alphas (alpha is at index 4)
            alpha_sum = curr_data[4] + prev_data[4]
            
            if dist < alpha_sum:
                # Keep the one with smaller alpha (more certified)
                if curr_data[4] < prev_data[4]:
                    coalesced[-1] = (curr_theta, curr_data)
            else:
                coalesced.append((curr_theta, curr_data))
    
    # build DataFrame
    df = pd.DataFrame(
        [(k,) + v for k, v in coalesced],
        columns=["theta", "pos", "neg", "zero", "perimeter", "alpha", "rot_num", "rot_den"])
    df.to_csv("p{}_N{}_orbits.csv".format(p, N), index=False)
    print("unique certified orbits:", len(df))
    
    # Generate metadata
    metadata = {}
    
    # Group by signature
    df['signature'] = df.apply(lambda row: (row['pos'], row['neg'], row['zero']), axis=1)
    signature_groups = df.groupby('signature')
    
    # Count orbits by signature
    metadata['orbit_counts_by_signature'] = {}
    for sig, group in signature_groups:
        sig_str = f"({sig[0]},{sig[1]},{sig[2]})"
        metadata['orbit_counts_by_signature'][sig_str] = len(group)
    
    # Perimeter statistics by signature
    metadata['perimeter_stats_by_signature'] = {}
    for sig, group in signature_groups:
        sig_str = f"({sig[0]},{sig[1]},{sig[2]})"
        stats = group['perimeter'].describe()
        metadata['perimeter_stats_by_signature'][sig_str] = {
            'count': int(stats['count']),
            'mean': float(stats['mean']),
            'std': float(stats['std']) if not pd.isna(stats['std']) else 0.0,
            'min': float(stats['min']),
            '25%': float(stats['25%']),
            '50%': float(stats['50%']),
            '75%': float(stats['75%']),
            'max': float(stats['max'])
        }
    
    # Rotation number statistics
    metadata['rotation_numbers'] = {}
    df['rotation'] = df.apply(lambda row: f"{int(row['rot_num'])}/{int(row['rot_den'])}", axis=1)
    rot_counts = df.groupby(['signature', 'rotation']).size()
    
    for (sig, rot), count in rot_counts.items():
        sig_str = f"({sig[0]},{sig[1]},{sig[2]})"
        if sig_str not in metadata['rotation_numbers']:
            metadata['rotation_numbers'][sig_str] = {}
        metadata['rotation_numbers'][sig_str][rot] = int(count)
    
    # Overall statistics
    metadata['overall'] = {
        'total_orbits': len(df),
        'unique_signatures': len(signature_groups),
        'p': float(p),
        'N': int(N),
        'N_SEEDS': int(N_SEEDS),
        'N_STEPS': int(N_STEPS)
    }
    
    # Write metadata to JSON file
    import json
    with open("p{}_N{}_metadata.json".format(p, N), 'w') as f:
        json.dump(metadata, f, indent=2)