import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Node:
    depth: int
    is_terminal: bool = False
    split_variables: List[int] = None
    coefficients: List[float] = None
    means: List[float] = None
    stds: List[float] = None
    split_point: float = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None

def expected_isolation_depth(m: int) -> float:
    if m <= 1:
        return 0
    return np.log2(m)

def fair_cut_tree(X: np.ndarray, p: int, d: int) -> Node:
    m, n = X.shape
    
    if m == 1:
        return Node(depth=d, is_terminal=True)
    
    if np.all(np.apply_along_axis(lambda col: np.unique(col).size == 1, axis=0, arr=X)):
        return Node(depth=d + expected_isolation_depth(m), is_terminal=True)

    z = np.zeros(m)
    
    split_vars = []
    coefficients = []
    means = []
    stds = []
    
    for _ in range(p):
        valid_vars = [i for i in range(n) if len(np.unique(X[:, i])) > 1]
        if not valid_vars:
            break
        v = np.random.choice(valid_vars)
        y = X[:, v]
        
        c = np.random.normal(0, 1)
        
        z += c * ((y - np.mean(y)) / np.std(y))
        
        split_vars.append(v)
        coefficients.append(c)
        means.append(np.mean(y))
        stds.append(np.std(y))
    
    best_split = None
    best_score = float('inf')
    for s in np.unique(z):
        z_left = z[z <= s]
        z_right = z[z > s]
        m_left, m_right = len(z_left), len(z_right)

        if m_left == 0 or m_right == 0:
            continue

        sigma_left = np.std(z_left) 
        sigma_right = np.std(z_right)

        score = (m_left * sigma_left + m_right * sigma_right) / (m_left + m_right)

        if score < best_score:
            best_score = score
            best_split = s
        
    if best_split is None:
        return Node(depth=d, is_terminal=True)
    
    X_left = X[z <= best_split]
    X_right = X[z > best_split]
    
    node = Node(
        depth=d,
        split_variables=split_vars,
        coefficients=coefficients,
        means=means,
        stds=stds,
        split_point=best_split
    )
    
    node.left = fair_cut_tree(X_left, p, d + 1)
    node.right = fair_cut_tree(X_right, p, d + 1)
    
    return node

def fair_cut_forest(X: np.ndarray, p: int, t: int, s: int) -> Tuple[List[Node], float]:
    m = X.shape[0]
    trees = []
    
    for _ in range(t):
        sample_indices = np.random.choice(m, size=s, replace=False)
        X_sample = X[sample_indices]
        
        tree = fair_cut_tree(X_sample, p, 0)
        trees.append(tree)
    
    q = expected_isolation_depth(m)
    return trees, q

def tree_score(x: np.ndarray, node: Node) -> float:
    if node.is_terminal:
        return node.depth
    
    z = 0
    for v, c, mean, std in zip(node.split_variables, node.coefficients, node.means, node.stds):
        z += c * ((x[v] - mean) / std)
    
    if z <= node.split_point:
        return tree_score(x, node.left)
    else:
        return tree_score(x, node.right)

def score_point(x: np.ndarray, trees: List[Node], q: float) -> float:
    d = 0
    for tree in trees:
        d += tree_score(x, tree)
    
    return 2 ** (-(d / len(trees)) / q)

def detect_anomalies(X: np.ndarray, p: int = 10, t: int = 100, s: int = 256) -> np.ndarray:
    trees, q = fair_cut_forest(X, p, t, s)
    
    scores = np.array([score_point(x, trees, q) for x in X])
    
    return scores