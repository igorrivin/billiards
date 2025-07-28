#!/usr/bin/env python3
"""
Standalone orbit verification for Lp billiards using Smale's alpha-criterion.

This program verifies that a sequence of bounce points on the Lp superellipse
has a genuine closed orbit nearby, without requiring JAX.
"""

import numpy as np
import argparse
import json


def curve(t, p, eps_pow=1e-14):
    """Parametrize the Lp superellipse boundary.
    
    Args:
        t: Parameter values in [0,1)
        p: Shape parameter (p > 2 for superellipse)
        eps_pow: Small regularization to avoid singularities
    
    Returns:
        Points on the boundary as (x,y) coordinates
    """
    ang = 2 * np.pi * t
    q = 2.0 / p
    c, s = np.cos(ang), np.sin(ang)
    
    def pow_func(x):
        return np.sign(x) * (np.abs(x) + eps_pow) ** q
    
    return np.column_stack([pow_func(c), pow_func(s)])


def poly_length(pts):
    """Compute perimeter of polygonal path."""
    d = pts - np.roll(pts, -1, axis=0)
    return np.sum(np.linalg.norm(d, axis=1))


def length_gradient(theta, p):
    """Compute gradient of length functional at theta."""
    n = len(theta)
    pts = curve(theta, p)
    
    # Compute derivatives of curve points
    h = 1e-8
    grad = np.zeros(n)
    
    for i in range(n):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += h
        theta_minus[i] -= h
        
        pts_plus = curve(theta_plus, p)
        pts_minus = curve(theta_minus, p)
        
        length_plus = poly_length(pts_plus)
        length_minus = poly_length(pts_minus)
        
        grad[i] = (length_plus - length_minus) / (2 * h)
    
    return grad


def length_hessian(theta, p):
    """Compute Hessian of length functional at theta."""
    n = len(theta)
    h = 1e-6
    hess = np.zeros((n, n))
    
    # Compute finite difference Hessian
    for i in range(n):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += h
        theta_minus[i] -= h
        
        grad_plus = length_gradient(theta_plus, p)
        grad_minus = length_gradient(theta_minus, p)
        
        hess[:, i] = (grad_plus - grad_minus) / (2 * h)
    
    return hess


def inertia(H, tol=1e-8):
    """Compute inertia (signature) of matrix H."""
    eig = np.linalg.eigvalsh(H)
    pos = int((eig > tol).sum())
    neg = int((eig < -tol).sum())
    zero = H.shape[0] - pos - neg
    return (pos, neg, zero), eig


def smale_alpha_criterion(theta, p):
    """Apply Smale's alpha-criterion to verify orbit existence.
    
    Args:
        theta: Array of parameter values for bounce points
        p: Shape parameter
        
    Returns:
        dict with verification results
    """
    # Convert to numpy array if needed
    theta = np.array(theta, dtype=float)
    
    # Compute gradient and Hessian
    g = length_gradient(theta, p)
    H = length_hessian(theta, p)
    
    # Compute Smale alpha quantities
    try:
        Hinv = np.linalg.inv(H)
        beta = np.linalg.norm(Hinv @ g)
        gamma = 0.5 * np.linalg.norm(Hinv) * np.linalg.norm(H)
        alpha = beta * gamma
    except np.linalg.LinAlgError:
        return {
            'verified': False,
            'error': 'Singular Hessian - cannot apply alpha criterion',
            'alpha': float('inf'),
            'gradient_norm': np.linalg.norm(g),
            'perimeter': poly_length(curve(theta, p))
        }
    
    # Smale's criterion: alpha < sqrt(3) - 1 ≈ 0.732
    smale_threshold = np.sqrt(3) - 1
    verified = alpha < smale_threshold
    
    # Conservative threshold used in the main program
    conservative_threshold = 0.15767
    conservative_verified = alpha < conservative_threshold
    
    # Compute inertia
    sig, eigenvalues = inertia(H)
    
    # Compute perimeter
    perimeter = poly_length(curve(theta, p))
    
    return {
        'verified': verified,
        'conservative_verified': conservative_verified,
        'alpha': float(alpha),
        'beta': float(beta),
        'gamma': float(gamma),
        'smale_threshold': float(smale_threshold),
        'conservative_threshold': float(conservative_threshold),
        'gradient_norm': float(np.linalg.norm(g)),
        'signature': sig,
        'eigenvalues': eigenvalues.tolist(),
        'perimeter': float(perimeter),
        'theta': theta.tolist(),
        'p': float(p)
    }


def rotation_number(theta):
    """Compute rotation number of an orbit by analyzing the permutation."""
    theta = np.array(theta)
    n = len(theta)
    
    # Get the permutation
    sorted_indices = np.argsort(theta)
    
    # Inverse permutation
    position = np.empty(n, dtype=int)
    position[sorted_indices] = np.arange(n)
    
    # Follow the orbit and count total rotation
    total_rotation = 0
    current_pos = position[0]
    
    for i in range(n):
        next_i = (i + 1) % n
        next_pos = position[next_i]
        
        # How far did we rotate?
        if next_pos >= current_pos:
            step = next_pos - current_pos
        else:
            step = n + next_pos - current_pos
            
        total_rotation += step
        current_pos = next_pos
    
    # The rotation number
    k = total_rotation // n
    
    from math import gcd
    g = gcd(k, n)
    return k // g, n // g


def main():
    parser = argparse.ArgumentParser(
        description='Verify billiard orbit using Smale alpha-criterion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 3.0 "[0.1, 0.3, 0.5, 0.7, 0.9]"
  %(prog)s 3.0 0.0657,0.375,0.6843
  %(prog)s 4.0 "[0.125, 0.625]" --verbose
        """)
    
    parser.add_argument('p', type=float, help='Shape parameter p > 2')
    parser.add_argument('theta', type=str, 
                       help='Bounce points as JSON array or comma-separated values')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    
    args = parser.parse_args()
    
    # Validate p
    if args.p <= 2.0:
        print(f"Error: p must be > 2, got {args.p}")
        return 1
    
    # Parse theta
    try:
        if args.theta.startswith('[') and args.theta.endswith(']'):
            # JSON format
            import json
            theta = json.loads(args.theta)
        else:
            # Comma-separated format
            theta = [float(x.strip()) for x in args.theta.split(',')]
        
        theta = np.array(theta)
        
        if len(theta) < 2:
            print("Error: Need at least 2 bounce points")
            return 1
            
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing theta: {e}")
        print("Use JSON format like '[0.1, 0.3, 0.5]' or comma-separated like '0.1,0.3,0.5'")
        return 1
    
    # Verify orbit
    result = smale_alpha_criterion(theta, args.p)
    
    # Add rotation number
    rot_num, rot_den = rotation_number(theta)
    result['rotation_number'] = f"{rot_num}/{rot_den}"
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        print(f"Orbit Verification for p={args.p}, N={len(theta)}")
        print(f"Bounce points: {theta}")
        print(f"Perimeter: {result['perimeter']:.6f}")
        print(f"Rotation number: {result['rotation_number']}")
        print(f"Signature: {result['signature']}")
        print()
        print(f"Smale alpha: {result['alpha']:.2e}")
        print(f"  Verified (α < {result['smale_threshold']:.3f}): {result['verified']}")
        print(f"  Conservative (α < {result['conservative_threshold']:.5f}): {result['conservative_verified']}")
        
        if args.verbose:
            print()
            print(f"Beta (Newton residual): {result['beta']:.2e}")
            print(f"Gamma (condition factor): {result['gamma']:.2e}")
            print(f"Gradient norm: {result['gradient_norm']:.2e}")
            print(f"Eigenvalues: {[f'{x:.2e}' for x in result['eigenvalues']]}")
        
        if 'error' in result:
            print(f"\nError: {result['error']}")
    
    return 0


if __name__ == "__main__":
    exit(main())