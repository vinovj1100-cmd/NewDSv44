"""Quantum AI Engine v4.4 — Enhanced with Quantum-Inspired Optimization.
Simulated Annealing, Ant Colony Optimization, Genetic Algorithm, 
Quantum Harmony Search, and Ensemble Router for warehouse pick-path optimization.
Includes: Predictive Congestion Engine, Dynamic Re-routing, SVG visualization.
"""
from __future__ import annotations
import hashlib, random, math, json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd

@dataclass
class ZoneNode:
    name: str; x: float; y: float
    velocity: str = "medium"
    priority: int = 5
    congestion: float = 0.0
    blocked: bool = False

@dataclass
class RouteSolution:
    nodes: List[ZoneNode]
    total_distance: float
    algorithm: str
    iterations: int
    convergence_history: List[float] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def node_names(self) -> List[str]:
        return [n.name for n in self.nodes]
    
    @property
    def efficiency_score(self) -> float:
        if not self.nodes or self.total_distance <= 0:
            return 0.0
        priority_bonus = sum((len(self.nodes) - i) * n.priority / 10.0 
                             for i, n in enumerate(self.nodes)) / max(1, len(self.nodes) * len(self.nodes) / 2)
        congestion_penalty = sum(n.congestion for n in self.nodes) / max(1, len(self.nodes))
        base = 100.0 / (1.0 + self.total_distance / max(1, len(self.nodes)))
        return round(base * (1.0 + priority_bonus * 0.3) * (1.0 - congestion_penalty * 0.4), 2)

def euclidean_distance(a: ZoneNode, b: ZoneNode) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def manhattan_distance(a: ZoneNode, b: ZoneNode) -> float:
    return abs(a.x - b.x) + abs(a.y - b.y)

def route_distance(nodes: List[ZoneNode], dist_fn: Callable = euclidean_distance) -> float:
    if len(nodes) < 2: return 0.0
    return sum(dist_fn(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))

def two_opt_swap(route: List[ZoneNode], i: int, j: int) -> List[ZoneNode]:
    return route[:i + 1] + route[i + 1:j + 1][::-1] + route[j + 1:]

def evaluate_route(nodes: List[ZoneNode], dist_fn: Callable = euclidean_distance,
                   priority_weight: float = 0.15, congestion_weight: float = 0.25) -> float:
    if len(nodes) < 2: return 0.0
    dist = route_distance(nodes, dist_fn)
    priority_penalty = sum(idx * (10 - node.priority) / 10.0 for idx, node in enumerate(nodes))
    congestion_penalty = sum(n.congestion * 50.0 for n in nodes)
    return dist + priority_weight * priority_penalty + congestion_weight * congestion_penalty

class SimulatedAnnealingTSP:
    def __init__(self, initial_temp: float = 1000.0, cooling_rate: float = 0.995,
                 min_temp: float = 1e-4, max_iterations: int = 10000,
                 dist_fn: Callable = euclidean_distance):
        self.initial_temp = initial_temp; self.cooling_rate = cooling_rate
        self.min_temp = min_temp; self.max_iterations = max_iterations; self.dist_fn = dist_fn
    
    def solve(self, nodes: List[ZoneNode]) -> RouteSolution:
        if len(nodes) <= 1:
            return RouteSolution(nodes, 0.0, "SA", 0)
        current = self._nearest_neighbor(nodes)
        current_cost = evaluate_route(current, self.dist_fn)
        best, best_cost = current[:], current_cost
        temp, convergence = self.initial_temp, []
        for iteration in range(self.max_iterations):
            if temp < self.min_temp: break
            i, j = random.sample(range(len(current)), 2)
            if i > j: i, j = j, i
            neighbor = two_opt_swap(current, i, j)
            neighbor_cost = evaluate_route(neighbor, self.dist_fn)
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current, current_cost = neighbor, neighbor_cost
                if current_cost < best_cost:
                    best, best_cost = current[:], current_cost
            temp *= self.cooling_rate
            convergence.append(best_cost)
        return RouteSolution(nodes=best, total_distance=route_distance(best, self.dist_fn),
                             algorithm="SimulatedAnnealing", iterations=len(convergence),
                             convergence_history=convergence[::max(1, len(convergence) // 100)],
                             metadata={"final_temp": temp, "best_cost": best_cost})
    
    @staticmethod
    def _nearest_neighbor(nodes: List[ZoneNode]) -> List[ZoneNode]:
        if not nodes: return []
        unvisited = set(range(1, len(nodes)))
        route = [nodes[0]]
        current = 0
        while unvisited:
            nearest = min(unvisited, key=lambda i: euclidean_distance(nodes[current], nodes[i]))
            route.append(nodes[nearest]); unvisited.remove(nearest); current = nearest
        return route

class AntColonyOptimizer:
    def __init__(self, n_ants: int = 20, n_iterations: int = 200,
                 alpha: float = 1.0, beta: float = 2.0, rho: float = 0.5,
                 Q: float = 100.0, dist_fn: Callable = euclidean_distance):
        self.n_ants = n_ants; self.n_iterations = n_iterations
        self.alpha = alpha; self.beta = beta; self.rho = rho; self.Q = Q; self.dist_fn = dist_fn
    
    def solve(self, nodes: List[ZoneNode]) -> RouteSolution:
        n = len(nodes)
        if n <= 1: return RouteSolution(nodes, 0.0, "ACO", 0)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j: dist_matrix[i, j] = max(self.dist_fn(nodes[i], nodes[j]), 0.001)
        heuristic = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    priority_boost = 1.0 + nodes[j].priority / 20.0
                    heuristic[i, j] = priority_boost / dist_matrix[i, j]
        pheromone = np.ones((n, n)) * 0.1
        best_route, best_cost, convergence = None, float("inf"), []
        for iteration in range(self.n_iterations):
            all_routes, all_costs = [], []
            for _ in range(self.n_ants):
                route = self._build_route(n, pheromone, heuristic)
                cost = self._route_cost(route, dist_matrix)
                all_routes.append(route); all_costs.append(cost)
                if cost < best_cost: best_cost = cost; best_route = route[:]
            pheromone *= (1 - self.rho)
            for route, cost in zip(all_routes, all_costs):
                deposit = self.Q / max(cost, 0.001)
                for i in range(len(route) - 1):
                    pheromone[route[i], route[i + 1]] += deposit
            convergence.append(best_cost)
        best_nodes = [nodes[i] for i in best_route]
        return RouteSolution(nodes=best_nodes, total_distance=route_distance(best_nodes, self.dist_fn),
                             algorithm="AntColonyOptimization", iterations=self.n_iterations,
                             convergence_history=convergence[::max(1, len(convergence) // 100)],
                             metadata={"pheromone_convergence": float(pheromone.mean())})
    
    def _build_route(self, n: int, pheromone: np.ndarray, heuristic: np.ndarray) -> List[int]:
        start = random.randrange(n)
        route, unvisited = [start], set(range(n)) - {start}
        while unvisited:
            current = route[-1]
            probs = np.zeros(n)
            for j in unvisited:
                probs[j] = (pheromone[current, j] ** self.alpha) * (heuristic[current, j] ** self.beta)
            total = probs.sum()
            if total == 0: next_node = random.choice(list(unvisited))
            else:
                probs = probs / total
                next_node = np.random.choice(n, p=probs)
                while next_node not in unvisited: next_node = np.random.choice(n, p=probs)
            route.append(next_node); unvisited.remove(next_node)
        return route
    
    def _route_cost(self, route: List[int], dist_matrix: np.ndarray) -> float:
        return sum(dist_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

class GeneticAlgorithmTSP:
    def __init__(self, population_size: int = 100, generations: int = 300,
                 mutation_rate: float = 0.15, crossover_rate: float = 0.8,
                 elite_size: int = 5, dist_fn: Callable = euclidean_distance):
        self.population_size = population_size; self.generations = generations
        self.mutation_rate = mutation_rate; self.crossover_rate = crossover_rate
        self.elite_size = elite_size; self.dist_fn = dist_fn
    
    def solve(self, nodes: List[ZoneNode]) -> RouteSolution:
        n = len(nodes)
        if n <= 1: return RouteSolution(nodes, 0.0, "GA", 0)
        population = [random.sample(range(n), n) for _ in range(self.population_size)]
        nn = SimulatedAnnealingTSP._nearest_neighbor(nodes)
        nn_indices = [nodes.index(node) for node in nn]
        population[0] = nn_indices
        best_route, best_cost, convergence = None, float("inf"), []
        for gen in range(self.generations):
            fitness = [(ind, evaluate_route([nodes[i] for i in ind], self.dist_fn)) for ind in population]
            fitness.sort(key=lambda x: x[1])
            if fitness[0][1] < best_cost: best_cost = fitness[0][1]; best_route = fitness[0][0][:]
            convergence.append(best_cost)
            new_pop = [ind[:] for ind, _ in fitness[:self.elite_size]]
            while len(new_pop) < self.population_size:
                p1 = self._tournament_select(fitness); p2 = self._tournament_select(fitness)
                child = self._ox_crossover(p1, p2) if random.random() < self.crossover_rate else p1[:]
                if random.random() < self.mutation_rate: child = self._mutate(child)
                new_pop.append(child)
            population = new_pop
        best_nodes = [nodes[i] for i in best_route]
        return RouteSolution(nodes=best_nodes, total_distance=route_distance(best_nodes, self.dist_fn),
                             algorithm="GeneticAlgorithm", iterations=self.generations,
                             convergence_history=convergence[::max(1, len(convergence) // 100)],
                             metadata={"final_fitness": best_cost})
    
    def _tournament_select(self, fitness, k: int = 3):
        contestants = random.sample(fitness, min(k, len(fitness)))
        contestants.sort(key=lambda x: x[1])
        return contestants[0][0]
    
    def _ox_crossover(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1); a, b = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[a:b + 1] = p1[a:b + 1]
        used = set(p1[a:b + 1])
        ptr = (b + 1) % n
        for gene in p2[b + 1:] + p2[:b + 1]:
            if gene not in used:
                child[ptr] = gene; used.add(gene); ptr = (ptr + 1) % n
        return child
    
    def _mutate(self, ind: List[int]) -> List[int]:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
        return ind

class QuantumHarmonySearch:
    def __init__(self, hms: int = 30, hmcr: float = 0.9, par: float = 0.3,
                 max_iterations: int = 5000, dist_fn: Callable = euclidean_distance):
        self.hms = hms; self.hmcr = hmcr; self.par = par; self.max_iterations = max_iterations
        self.dist_fn = dist_fn
    
    def solve(self, nodes: List[ZoneNode]) -> RouteSolution:
        n = len(nodes)
        if n <= 1: return RouteSolution(nodes, 0.0, "QHS", 0)
        hm = [random.sample(range(n), n) for _ in range(self.hms)]
        costs = [evaluate_route([nodes[i] for i in h], self.dist_fn) for h in hm]
        sorted_indices = np.argsort(costs)
        hm = [hm[i] for i in sorted_indices]; costs = [costs[i] for i in sorted_indices]
        best, best_cost, convergence = hm[0][:], costs[0], []
        for _ in range(self.max_iterations):
            new_harmony, used = [], set()
            for pos in range(n):
                if random.random() < self.hmcr and hm:
                    chosen = random.choice(hm)[pos]
                    if chosen not in used: new_harmony.append(chosen); used.add(chosen)
                    else:
                        available = [i for i in range(n) if i not in used]
                        if available: new_harmony.append(random.choice(available)); used.add(new_harmony[-1])
                else:
                    available = [i for i in range(n) if i not in used]
                    if available: new_harmony.append(random.choice(available)); used.add(new_harmony[-1])
            for i in range(n):
                if i not in used:
                    for j in range(n):
                        if len(new_harmony) <= j or new_harmony[j] in used:
                            if len(new_harmony) <= j: new_harmony.append(i)
                            else: new_harmony[j] = i
                            used.add(i); break
            if len(new_harmony) == n and len(set(new_harmony)) == n:
                new_cost = evaluate_route([nodes[i] for i in new_harmony], self.dist_fn)
                if new_cost < costs[-1]:
                    hm[-1] = new_harmony; costs[-1] = new_cost
                    sorted_indices = np.argsort(costs)
                    hm = [hm[i] for i in sorted_indices]; costs = [costs[i] for i in sorted_indices]
                    if costs[0] < best_cost: best = hm[0][:]; best_cost = costs[0]
            convergence.append(best_cost)
        best_nodes = [nodes[i] for i in best]
        return RouteSolution(nodes=best_nodes, total_distance=route_distance(best_nodes, self.dist_fn),
                             algorithm="QuantumHarmonySearch", iterations=len(convergence),
                             convergence_history=convergence[::max(1, len(convergence) // 100)],
                             metadata={"hmcr": self.hmcr, "par": self.par})

class QuantumEnsembleRouter:
    ALGORITHMS = {"sa": SimulatedAnnealingTSP, "aco": AntColonyOptimizer,
                  "ga": GeneticAlgorithmTSP, "qhs": QuantumHarmonySearch}
    
    def __init__(self, grid_w: int = 24, grid_h: int = 16):
        self.grid_w = grid_w; self.grid_h = grid_h
        self.zones: Dict[str, ZoneNode] = {}
        self.heat_map = np.zeros((grid_h, grid_w))
        self.visit_log = deque(maxlen=500)
        self._algorithm_instances: Dict[str, object] = {}
    
    def init_zones_from_skus(self, sku_list: List[str], sku_locations: Optional[Dict[str, str]] = None):
        self.zones = {}
        for sku in sku_list:
            h = hashlib.md5(sku.encode()).hexdigest()
            x = int(h[:2], 16) % (self.grid_w - 3) + 2
            y = int(h[2:4], 16) % (self.grid_h - 3) + 2
            velocity = ["high", "medium", "low"][int(h[4:6], 16) % 3]
            priority = max(1, min(10, int(h[6:8], 16) % 10 + 1))
            zone_name = sku_locations.get(sku, f"Z-{x}-{y}") if sku_locations else f"Z-{x}-{y}"
            self.zones[sku] = ZoneNode(name=zone_name, x=float(x), y=float(y),
                                        velocity=velocity, priority=priority)
    
    def update_congestion(self, zone_name: str, congestion: float):
        for node in self.zones.values():
            if node.name == zone_name:
                node.congestion = max(0.0, min(1.0, congestion)); break
    
    def block_zone(self, zone_name: str, blocked: bool = True):
        for node in self.zones.values():
            if node.name == zone_name:
                node.blocked = blocked; break
    
    def optimize(self, sku_list: List[str], algorithm: str = "ensemble",
                 sku_locations: Optional[Dict[str, str]] = None,
                 time_budget_sec: float = 2.0) -> RouteSolution:
        self.init_zones_from_skus(sku_list, sku_locations)
        nodes = list(self.zones.values())
        active_nodes = [n for n in nodes if not n.blocked]
        if len(active_nodes) < 2:
            return RouteSolution(active_nodes, 0.0, "None", 0,
                               metadata={"error": "Insufficient active zones"})
        if algorithm == "ensemble":
            return self._run_ensemble(active_nodes, time_budget_sec)
        elif algorithm in self.ALGORITHMS:
            solver = self.ALGORITHMS[algorithm]()
            return solver.solve(active_nodes)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _run_ensemble(self, nodes: List[ZoneNode], time_budget_sec: float) -> RouteSolution:
        solutions = []
        for name, cls in self.ALGORITHMS.items():
            try:
                solver = cls()
                sol = solver.solve(nodes)
                solutions.append(sol)
            except Exception: continue
        if not solutions:
            nn = SimulatedAnnealingTSP._nearest_neighbor(nodes)
            return RouteSolution(nn, route_distance(nn), "Fallback-NN", 0,
                                 metadata={"note": "All advanced algorithms failed"})
        best = max(solutions, key=lambda s: s.efficiency_score)
        best.metadata["ensemble_candidates"] = len(solutions)
        best.metadata["algorithms_tried"] = [s.algorithm for s in solutions]
        best.metadata["all_scores"] = {s.algorithm: s.efficiency_score for s in solutions}
        return best
    
    def dynamic_reroute(self, current_route: RouteSolution, blocked_zone: str) -> RouteSolution:
        self.block_zone(blocked_zone, True)
        remaining_skus = [n.name for n in current_route.nodes if n.name != blocked_zone]
        if len(remaining_skus) < 2: return current_route
        return self.optimize(remaining_skus, algorithm="sa")
    
    def generate_svg(self, solution: RouteSolution, width: int = 900, height: int = 550) -> str:
        cw, ch = width / self.grid_w, height / self.grid_h
        mx = self.heat_map.max() or 1
        svg_parts = [
            f'<svg width="{width}" height="{height}" style="background:#050a19; border-radius:12px;">',
            '<defs>',
            '  <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">',
            '    <feGaussianBlur stdDeviation="3" result="blur"/>',
            '    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>',
            '  </filter>',
            '  <linearGradient id="routeGrad" x1="0%" y1="0%" x2="100%" y2="0%">',
            '    <stop offset="0%" style="stop-color:#00ff88;stop-opacity:1" />',
            '    <stop offset="50%" style="stop-color:#64ffda;stop-opacity:1" />',
            '    <stop offset="100%" style="stop-color:#ff6b6b;stop-opacity:1" />',
            '  </linearGradient>',
            '</defs>',
        ]
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                heat = self.heat_map[y, x] / mx
                if heat > 0.03:
                    r = int(255 * min(1, heat * 2))
                    g = int(100 + 155 * max(0, 1 - heat))
                    b = int(100 + 100 * max(0, 1 - heat * 1.5))
                    svg_parts.append(f'<rect x="{x*cw:.1f}" y="{y*ch:.1f}" width="{cw:.1f}" height="{ch:.1f}" fill="rgba({r},{g},{b},{heat*0.4})"/>')
        for i in range(self.grid_w + 1):
            svg_parts.append(f'<line x1="{i*cw}" y1="0" x2="{i*cw}" y2="{height}" stroke="rgba(100,255,218,0.08)" stroke-width="0.5"/>')
        for i in range(self.grid_h + 1):
            svg_parts.append(f'<line x1="0" y1="{i*ch}" x2="{width}" y2="{i*ch}" stroke="rgba(100,255,218,0.08)" stroke-width="0.5"/>')
        for node in self.zones.values():
            cx, cy = node.x * cw, node.y * ch
            c = {"high": "#64ffda", "medium": "#00b4db", "low": "#8892b0"}.get(node.velocity, "#8892b0")
            opacity = "0.3" if node.blocked else "0.6"
            svg_parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="10" fill="{c}" opacity="{opacity}" filter="url(#glow)"/>'
                f'<text x="{cx}" y="{cy-15}" fill="#ccd6f6" font-size="9" text-anchor="middle" font-family="monospace">{node.name}</text>'
            )
        if solution and len(solution.nodes) > 1:
            pts = " ".join([f'{n.x*cw:.1f},{n.y*ch:.1f}' for n in solution.nodes])
            svg_parts.append(
                f'<polyline points="{pts}" fill="none" stroke="url(#routeGrad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.9" filter="url(#glow)"/>'
            )
            sx, sy = solution.nodes[0].x * cw, solution.nodes[0].y * ch
            svg_parts.append(
                f'<circle cx="{sx}" cy="{sy}" r="8" fill="#00ff88" filter="url(#glow)"/>'
                f'<text x="{sx}" y="{sy+20}" fill="#00ff88" font-size="10" text-anchor="middle" font-weight="bold">START</text>'
            )
            for i, n in enumerate(solution.nodes[1:], 1):
                cx, cy = n.x * cw, n.y * ch
                c = "#ff6b6b" if i == len(solution.nodes) - 1 else "#ffd93d"
                svg_parts.append(
                    f'<circle cx="{cx}" cy="{cy}" r="6" fill="{c}"/>'
                    f'<text x="{cx}" y="{cy+22}" fill="#fff" font-size="9" text-anchor="middle" font-family="monospace">{i}. {n.name[:10]}</text>'
                )
        svg_parts.append(
            f'<rect x="{width-200}" y="10" width="190" height="90" rx="8" fill="rgba(10,20,40,0.85)" stroke="rgba(100,255,218,0.2)"/>'
            f'<text x="{width-190}" y="30" fill="#64ffda" font-size="10" font-weight="bold">Quantum Route</text>'
            f'<text x="{width-190}" y="48" fill="#8892b0" font-size="9">Algo: {solution.algorithm}</text>'
            f'<text x="{width-190}" y="64" fill="#8892b0" font-size="9">Dist: {solution.total_distance:.1f}</text>'
            f'<text x="{width-190}" y="80" fill="#8892b0" font-size="9">Efficiency: {solution.efficiency_score:.0f}%</text>'
        )
        svg_parts.append('</svg>')
        return "\n".join(svg_parts)
    
    def convergence_chart_data(self, solution: RouteSolution) -> pd.DataFrame:
        if not solution.convergence_history: return pd.DataFrame()
        return pd.DataFrame({"iteration": range(len(solution.convergence_history)),
                             "best_cost": solution.convergence_history})

class PredictiveCongestionEngine:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.history: Dict[str, deque] = {}
    
    def record_visit(self, zone: str, timestamp: Optional[datetime] = None):
        ts = timestamp or datetime.now()
        if zone not in self.history: self.history[zone] = deque(maxlen=100)
        self.history[zone].append(ts)
    
    def predict_congestion(self, zone: str, horizon_minutes: int = 30) -> float:
        if zone not in self.history or len(self.history[zone]) < 3: return 0.0
        visits = list(self.history[zone])
        intervals = [(visits[i] - visits[i - 1]).total_seconds() / 60.0 for i in range(1, len(visits))]
        if not intervals: return 0.0
        avg_interval = sum(intervals) / len(intervals)
        predicted_visits = horizon_minutes / max(avg_interval, 1.0)
        return min(1.0, predicted_visits / 10.0)
    
    def get_all_predictions(self, horizon_minutes: int = 30) -> Dict[str, float]:
        return {z: self.predict_congestion(z, horizon_minutes) for z in self.history}

class QuantumRouteOptimizer:
    def __init__(self, grid_w: int = 24, grid_h: int = 16):
        self.router = QuantumEnsembleRouter(grid_w, grid_h)
        self.grid_w = grid_w; self.grid_h = grid_h
        self.zones = self.router.zones
        self.heat_map = self.router.heat_map
        self.visit_log = self.router.visit_log
        self._init_zones()  # FIX: ensure zones are populated on startup
    
    def _init_zones(self):
        zones = {}
        for l in "ABCDEFGH":
            for n in range(1, 4):
                h = hashlib.md5(f"{l}{n}".encode()).hexdigest()
                zones[f"{l}{n}"] = {"x": int(h[:2], 16) % (self.grid_w - 3) + 2,
                                     "y": int(h[2:4], 16) % (self.grid_h - 3) + 2,
                                     "velocity": random.choice(["high", "medium", "low"])}
        self.zones = zones
        self.router.zones = {name: ZoneNode(name=name, x=float(z["x"]), y=float(z["y"]), velocity=z["velocity"])
                             for name, z in zones.items()}
        return zones
    
    def update_heat(self, sku_visits: Dict[str, int]):
        for sku, visits in sku_visits.items():
            h = hashlib.md5(sku.encode()).hexdigest()
            zone_keys = list(self.zones.keys())
            z_name = zone_keys[int(h, 16) % len(zone_keys)]
            z = self.zones[z_name]
            self.heat_map[int(z["y"]), int(z["x"])] += visits
            self.visit_log.append({"sku": sku, "zone": z_name, "visits": visits})
    
    def optimize_route(self, sku_list: List[str]) -> List[Dict]:
        if not sku_list: return []
        solution = self.router.optimize(sku_list, algorithm="sa")
        return [{"sku": n.name, "zone": n.name, "x": n.x, "y": n.y, "priority": n.priority}
                for n in solution.nodes]
    
    def _dist(self, a: Dict, b: Dict) -> float:
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])
    
    def generate_svg(self, route: List[Dict], width: int = 900, height: int = 550) -> str:
        if route:
            nodes = [ZoneNode(name=r.get("sku", r.get("zone", "?")), x=r.get("x", 0), y=r.get("y", 0))
                     for r in route]
            sol = RouteSolution(nodes=nodes,
                                total_distance=sum(self._dist(route[i], route[i + 1]) for i in range(len(route) - 1)) if len(route) > 1 else 0,
                                algorithm="Legacy", iterations=0)
        else:
            sol = RouteSolution([], 0.0, "Legacy", 0)
        return self.router.generate_svg(sol, width, height)
