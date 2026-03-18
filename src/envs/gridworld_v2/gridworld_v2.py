"""
Improved Gridworld v2 — Bridging the gap between the simple 5x5 grid
and the highway-env intersection-v1 environment.

Key improvements over v1:
    - 7x7 grid with a cross-shaped intersection (two perpendicular roads)
    - 3 discrete speed actions: STOP (0), SLOW (1), FAST (2)
      Mirrors intersection-v1's target_speeds: [0, 4.5, 9]
    - Multiple other vehicles with stochastic spawning
    - IDM-like reactive behavior for other vehicles:
      * Desired speed = FAST
      * Decelerate when approaching ego near conflict zone
      * Only non-priority vehicles react (ego's road has lower priority)
    - Progress-based r_self: speed reward + collision penalty + arrival bonus
    - Reaction-based r_global: measures disruption ego causes to others
      * Baseline when no disruption, penalty when others brake/wait
      * Flow bonus when ego clears the intersection
    - Collision detection at the conflict point

Design choices aligned with intersection-v1:
    - Ego travels West→East (row 3), others travel South→North (col 3)
    - Conflict point at (3, 3)
    - Others have priority (like cross-traffic at an unsignalized intersection)
    - Episode ends when ego reaches east edge OR collision OR timeout

State space: (ego_x, ego_speed, other1_y, other1_speed, other2_y, other2_speed)
    With 7 positions × 3 speeds × (8 positions × 3 speeds)^2 = ~127,000 states
    (position 7 = "not spawned yet" for others)
    Still tractable for value iteration.

References:
    - Treiber, Hennecke & Helbing (2000) "Congested traffic states in
      empirical observations and microscopic simulations" — IDM
    - Schwarting et al. (2019) "Social behavior for autonomous vehicles" — SVO
    - Intersection-v1 config: target_speeds=[0, 4.5, 9], 3 actions, longitudinal only
"""

import numpy as np

# Minimal gym-like interfaces so the file runs standalone without gymnasium.
# When used in the real project, replace with `import gymnasium as gym` etc.
try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            self.shape = self.low.shape if shape is None else shape
            self.dtype = dtype

    class _Discrete:
        def __init__(self, n):
            self.n = n

    class spaces:
        Box = _Box
        Discrete = _Discrete

    class _BaseEnv:
        metadata = {}
        np_random = np.random.default_rng()
        def reset(self, seed=None, options=None):
            if seed is not None:
                self.np_random = np.random.default_rng(seed)

    gym = None

    class _BaseEnv:
        metadata = {}
        np_random = np.random.default_rng()
        def reset(self, seed=None, options=None):
            if seed is not None:
                self.np_random = np.random.default_rng(seed)

_GymEnv = gym.Env if _HAS_GYM else _BaseEnv


# ---------------------------------------------------------------------------
# Constants — aligned with intersection-v1 where possible
# ---------------------------------------------------------------------------
GRID_SIZE = 7
CONFLICT_X = 3          # Ego's x-coordinate at the intersection center
CONFLICT_Y = 3          # Other's y-coordinate at the intersection center
EGO_ROW = 3             # Ego travels along y=3 (West→East)
OTHER_COL = 3           # Others travel along x=3 (South→North)

# Speed levels (cells per step)
SPEED_STOP = 0
SPEED_SLOW = 1
SPEED_FAST = 2
NUM_SPEEDS = 3

# Other vehicle states
NOT_SPAWNED = -1         # Sentinel: vehicle hasn't appeared yet
MAX_OTHERS = 2           # Number of other vehicle slots

# Spawn probability per step (when a slot is empty)
# Intersection-v1 uses spawn_probability=0.3
SPAWN_PROB = 0.3

# Reward weights — designed for SVO separability
# r_self components
W_SPEED = 0.4           # Continuous speed reward (per step)
W_COLLISION = -5.0      # Collision penalty
W_ARRIVED = 2.0         # Terminal arrival bonus (smaller — step costs must bite)
W_STEP = -0.3           # Per-step cost (accumulates: 25 steps × -0.3 = -7.5)

# r_global components
R_BASELINE = 0.0        # Per-step baseline when ego causes no disruption
W_DELAY = -1.0          # Per speed-level of forced deceleration
W_BLOCK = -1.5          # Other is completely stopped because of ego
W_OTHER_COLLISION = -5.0  # Ego caused a neighbor crash
W_FLOW = 1.5            # Bonus when ego clears the intersection (one-time)
W_OTHER_ARRIVED = 1.0   # Bonus when an other vehicle arrives at its goal
W_EGO_CLEAR = 1.0       # r_global bonus when ego arrives (frees road capacity)

# IDM-like reaction zone: other vehicles start reacting when ego is within
# this many cells of the conflict point AND on a collision course.
REACTION_DISTANCE = 3

MAX_STEPS = 25


class SVOIntersectionGridV2(_GymEnv):
    """
    A 7x7 gridworld intersection with IDM-like reactive traffic,
    3-speed actions, and SVO-compatible reward decomposition.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode=None, deterministic_traffic=False):
        """
        Args:
            render_mode: "console" for text rendering, None for headless.
            deterministic_traffic: If True, always spawn both others at
                fixed positions (useful for value iteration).
        """
        super().__init__()
        self.grid_size = GRID_SIZE
        self.render_mode = render_mode
        self.deterministic_traffic = deterministic_traffic

        # Action space: 0=STOP, 1=SLOW, 2=FAST
        self.action_space = spaces.Discrete(3)

        # Observation: [ego_x, ego_speed, other1_y, other1_speed, other2_y, other2_speed]
        # Positions in [0, grid_size-1], speeds in [0, 2]
        # Other y = -1 means not spawned
        low = np.array([0, 0, -1, 0, -1, 0], dtype=np.float32)
        high = np.array([GRID_SIZE-1, NUM_SPEEDS-1,
                         GRID_SIZE-1, NUM_SPEEDS-1,
                         GRID_SIZE-1, NUM_SPEEDS-1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Ego starts at west edge, speed=FAST
        self.ego_x = 0
        self.ego_speed = SPEED_FAST

        # Other vehicles: [y_position, speed]
        # In deterministic mode, both spawn at fixed positions
        if self.deterministic_traffic:
            self.others = [
                [0, SPEED_FAST],     # Other 1: starts at bottom
                [NOT_SPAWNED, SPEED_FAST],  # Other 2: not yet spawned
            ]
        else:
            self.others = [
                [0, SPEED_FAST],     # Other 1 always starts
                [NOT_SPAWNED, SPEED_FAST],  # Other 2 may spawn later
            ]

        self.step_count = 0
        self.ego_arrived = False
        self.ego_crashed = False
        self.ego_cleared_intersection = False  # Track if ego passed conflict

        # Track previous speeds of others for reaction measurement
        self._prev_other_speeds = [o[1] for o in self.others]
        self._post_reaction_speeds = [o[1] for o in self.others]

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, action):
        self.step_count += 1

        # Store pre-step state
        prev_ego_x = self.ego_x
        prev_ego_speed = self.ego_speed
        self._prev_other_speeds = [o[1] for o in self.others]
        prev_others = [[o[0], o[1]] for o in self.others]

        # --- 1. Spawning (initial spawn for not-yet-spawned vehicles) ---
        for i, other in enumerate(self.others):
            if other[0] == NOT_SPAWNED:
                if self.deterministic_traffic:
                    if i == 0:
                        other[0] = 0
                        other[1] = SPEED_FAST
                    elif self.others[0][0] >= 2 and self.step_count >= 3:
                        other[0] = 0
                        other[1] = SPEED_FAST
                else:
                    if i == 0:
                        other[0] = 0
                        other[1] = SPEED_FAST
                    elif self.np_random.random() < SPAWN_PROB:
                        other[0] = 0
                        other[1] = SPEED_FAST

        # --- 2. Compute other vehicles' IDM-like reactions ---
        self._update_others_reactive(action)
        self._post_reaction_speeds = [o[1] for o in self.others]

        # --- 3. Move ego ---
        self.ego_speed = action  # Direct speed control (like DiscreteMetaAction)
        next_ego_x = self.ego_x + self.ego_speed
        next_ego_x = min(next_ego_x, self.grid_size - 1)
        self.ego_x = next_ego_x

        # --- 4. Move others (after reaction computed) ---
        other_arrivals = self._move_others()

        # --- 4b. Respawn arrived others (continuous traffic) ---
        self._maybe_spawn()

        # --- 5. Collision detection ---
        collision, collided_other_idx = self._check_collisions()
        if collision:
            self.ego_crashed = True

        # --- 6. Check if ego cleared intersection ---
        if prev_ego_x <= CONFLICT_X and self.ego_x > CONFLICT_X:
            self.ego_cleared_intersection = True

        # --- 7. Check arrival ---
        self.ego_arrived = (self.ego_x >= self.grid_size - 1)

        # --- 8. Compute rewards ---
        r_self = self._compute_r_self(prev_ego_x)
        r_global = self._compute_r_global(prev_others, collided_other_idx,
                                           other_arrivals)

        # --- 9. Termination ---
        terminated = self.ego_arrived or self.ego_crashed
        truncated = (self.step_count >= MAX_STEPS) and not terminated

        reward = r_self  # Default reward is r_self; SVO combination done externally

        info = {
            'rewards/component_self': float(r_self),
            'rewards/component_global': float(r_global),
            'collision': self.ego_crashed,
            'ego_arrived': self.ego_arrived,
            'ego_speed': self.ego_speed,
            'ego_x': self.ego_x,
            'others': [[o[0], o[1]] for o in self.others],
            'step': self.step_count,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # IDM-like reactive behavior for other vehicles
    # ------------------------------------------------------------------

    def _update_others_reactive(self, ego_action):
        """
        Update other vehicles' speeds based on ego's proximity to the
        conflict point. Mimics IDM: others have desired speed = FAST,
        but decelerate when ego threatens to occupy the conflict zone.

        Others have priority on the north-south road, so they only react
        (brake) when ego is actually going to be at the conflict point
        at the same time. This avoids deadlocks — the other doesn't
        preemptively stop, only brakes when a collision is imminent.
        """
        ego_intended_speed = ego_action
        ego_next_x = min(self.ego_x + ego_intended_speed, self.grid_size - 1)

        for i, other in enumerate(self.others):
            if other[0] == NOT_SPAWNED:
                continue

            # Other's desired speed is FAST
            desired_speed = SPEED_FAST

            # Check if ego will be at or passing through conflict zone
            ego_at_conflict = (
                self.ego_x <= CONFLICT_X <= ego_next_x  # Ego crosses/reaches conflict
                or self.ego_x == CONFLICT_X              # Ego is already at conflict
            )

            # Check if other is approaching conflict zone
            other_dist_to_conflict = CONFLICT_Y - other[0]

            if ego_at_conflict and 0 < other_dist_to_conflict <= REACTION_DISTANCE:
                # IDM-like proportional braking:
                # Close to conflict + ego present → brake harder
                if other_dist_to_conflict == 1:
                    # Imminent collision: full stop
                    desired_speed = SPEED_STOP
                elif other_dist_to_conflict == 2:
                    # Close: slow down
                    desired_speed = SPEED_SLOW
                else:
                    # Far but aware: slight deceleration
                    desired_speed = SPEED_SLOW

            # Also brake if ego is sitting ON the conflict point (blocking)
            if self.ego_x == CONFLICT_X and other_dist_to_conflict > 0:
                if other_dist_to_conflict <= 2:
                    desired_speed = SPEED_STOP
                else:
                    desired_speed = min(desired_speed, SPEED_SLOW)

            other[1] = desired_speed

    def _move_others(self):
        """Move other vehicles according to their (already computed) speeds.
        Returns list of booleans indicating which others just arrived."""
        arrivals = [False] * len(self.others)
        for i, other in enumerate(self.others):
            if other[0] == NOT_SPAWNED:
                continue

            # Already at goal before move — will be respawned by _maybe_spawn
            if other[0] >= self.grid_size - 1:
                other[1] = SPEED_STOP
                continue

            prev_y = other[0]
            next_y = other[0] + other[1]
            next_y = min(next_y, self.grid_size - 1)
            other[0] = next_y

            # Track if this other just arrived
            if prev_y < self.grid_size - 1 and next_y >= self.grid_size - 1:
                arrivals[i] = True

        return arrivals

    def _maybe_spawn(self):
        """Spawn / respawn vehicles. Arrived vehicles respawn at y=0,
        creating a continuous traffic stream like a real intersection."""
        if self.deterministic_traffic:
            for i, other in enumerate(self.others):
                if other[0] == NOT_SPAWNED:
                    # Initial spawn: other1 always, other2 after other1 progresses
                    if i == 0:
                        other[0] = 0
                        other[1] = SPEED_FAST
                    elif self.others[0][0] >= 2 and self.step_count >= 3:
                        other[0] = 0
                        other[1] = SPEED_FAST
                elif other[0] >= self.grid_size - 1:
                    # Arrived → respawn at bottom
                    other[0] = 0
                    other[1] = SPEED_FAST
            return

        for i, other in enumerate(self.others):
            if other[0] == NOT_SPAWNED:
                # First vehicle always spawns; others stochastic
                if i == 0:
                    other[0] = 0
                    other[1] = SPEED_FAST
                elif self.np_random.random() < SPAWN_PROB:
                    other[0] = 0
                    other[1] = SPEED_FAST
            elif other[0] >= self.grid_size - 1:
                # Arrived → respawn stochastically
                if self.np_random.random() < SPAWN_PROB:
                    other[0] = 0
                    other[1] = SPEED_FAST

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def _check_collisions(self):
        """
        Check if ego and any other vehicle occupy the conflict point
        simultaneously, or if ego moved through conflict while other
        was there.
        """
        collided = False
        collided_idx = -1

        for i, other in enumerate(self.others):
            if other[0] == NOT_SPAWNED:
                continue

            # Both at conflict point
            if self.ego_x == CONFLICT_X and other[0] == CONFLICT_Y:
                collided = True
                collided_idx = i
                break

            # Ego at conflict and other at conflict (same cell)
            if (self.ego_x == OTHER_COL and
                    EGO_ROW == other[0] and
                    self.ego_x == CONFLICT_X):
                collided = True
                collided_idx = i
                break

        return collided, collided_idx

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_r_self(self, prev_ego_x):
        """
        Ego's individual reward. Components:
        - Speed reward: proportional to ego speed (encourages progress)
        - Collision penalty: large negative on crash
        - Arrival bonus: one-time reward for reaching the goal
        - Step cost: small negative per step (discourages waiting forever)
        """
        if self.ego_crashed:
            return W_COLLISION

        r = W_STEP  # Per-step cost

        # Speed reward: normalized to [0, 1]
        speed_normalized = self.ego_speed / SPEED_FAST
        r += W_SPEED * speed_normalized

        # Arrival bonus (one-time)
        if self.ego_arrived:
            r += W_ARRIVED

        return r

    def _compute_r_global(self, prev_others, collided_other_idx, other_arrivals):
        """
        Social/global reward measuring ego's impact on others. Components:
        - Baseline: positive when ego causes no disruption
        - Delay cost: negative proportional to how much others decelerated
        - Block cost: extra penalty when others are completely stopped by ego
        - Collision cost: large negative if ego caused a neighbor crash
        - Flow bonus: positive when ego clears the intersection
        - Other arrival bonus: positive when another vehicle reaches its goal

        Design principle: when ego drives carefully (moderate speed, good timing),
        r_global stays near baseline. When ego is aggressive, others brake hard
        (low r_global). When ego waits forever, r_global stays at baseline but
        r_self drops. The sweet spot for prosocial agents is timing the crossing
        to minimize disruption while still making progress.
        """
        # Collision with other vehicle
        if collided_other_idx >= 0:
            return W_OTHER_COLLISION

        r = 0.0

        # Social components only apply when ego is in the influence zone
        ego_in_influence_zone = (CONFLICT_X - 1 <= self.ego_x <= CONFLICT_X)

        if ego_in_influence_zone:
            # Count active others from prev state
            active_others = 0
            for i in range(len(self.others)):
                po = prev_others[i]
                if po[0] != NOT_SPAWNED and po[0] < self.grid_size - 1:
                    active_others += 1

            if active_others > 0:
                r += R_BASELINE

            # Other arrival bonuses
            for i, arrived in enumerate(other_arrivals):
                if arrived:
                    r += W_OTHER_ARRIVED

            # Disruption: check post-reaction speeds of active, non-arrived others
            for i in range(len(self.others)):
                po = prev_others[i]
                if po[0] == NOT_SPAWNED or po[0] >= self.grid_size - 1:
                    continue
                if other_arrivals[i]:
                    continue

                # Use the speed that was set during _update_others_reactive
                # After reaction + move + respawn, we need the reaction speed
                # which was stored in other[1] before _move_others changed position
                # Since we can't easily get that, use a simpler approach:
                # the prev_other_speeds were captured BEFORE reactions in step(),
                # but we need post-reaction speed. Let's store it.
                curr_speed = self._post_reaction_speeds[i]
                speed_loss = SPEED_FAST - curr_speed
                if speed_loss > 0:
                    if curr_speed == SPEED_STOP:
                        r += W_BLOCK
                    else:
                        r += W_DELAY * (speed_loss / SPEED_FAST)

        # Flow bonus: ego cleared the intersection, freeing capacity
        if self.ego_cleared_intersection:
            r += W_FLOW
            self.ego_cleared_intersection = False  # One-time bonus

        # Ego arrival bonus: ego leaving the road frees capacity for everyone
        if self.ego_arrived:
            r += W_EGO_CLEAR

        return r

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        obs = np.array([
            self.ego_x,
            self.ego_speed,
            self.others[0][0],
            self.others[0][1],
            self.others[1][0],
            self.others[1][1],
        ], dtype=np.float32)
        return obs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode != "console":
            return

        grid = np.full((self.grid_size, self.grid_size), ".")

        # Draw roads
        for x in range(self.grid_size):
            grid[EGO_ROW, x] = "-"  # East-west road
        for y in range(self.grid_size):
            grid[y, OTHER_COL] = "|"  # North-south road
        grid[CONFLICT_Y, CONFLICT_X] = "+"  # Intersection

        # Draw other vehicles
        for i, other in enumerate(self.others):
            if other[0] == NOT_SPAWNED or other[0] >= self.grid_size:
                continue
            y = int(other[0])
            label = str(i + 1)
            speed_char = ["s", "m", "f"][other[1]]  # stop/medium/fast
            if 0 <= y < self.grid_size:
                grid[y, OTHER_COL] = f"{label}"

        # Draw ego
        ex = int(self.ego_x)
        if 0 <= ex < self.grid_size:
            speed_char = ["S", "M", "F"][self.ego_speed]
            if grid[EGO_ROW, ex] in ["1", "2"]:
                grid[EGO_ROW, ex] = "X"  # Collision
            else:
                grid[EGO_ROW, ex] = "E"

        print(f"\nStep: {self.step_count}  Ego speed: {self.ego_speed}  "
              f"Crashed: {self.ego_crashed}  Arrived: {self.ego_arrived}")
        # Print top to bottom (row 6 at top, row 0 at bottom)
        for row in reversed(range(self.grid_size)):
            print(" ".join(grid[row]))

        # Print other vehicle info
        for i, other in enumerate(self.others):
            status = "not spawned" if other[0] == NOT_SPAWNED else f"y={other[0]} spd={other[1]}"
            print(f"  Other {i+1}: {status}")
        print("-" * 20)


# =========================================================================
# Value Iteration solver (deterministic version for expert generation)
# =========================================================================

class SVOValueIterationV2:
    """
    Value Iteration solver for SVOIntersectionGridV2.

    Uses the deterministic variant of the environment for exact solving.
    State: (ego_x, ego_speed, other1_y, other1_speed, other2_y, other2_speed)

    For tractability with value iteration, we use deterministic traffic:
    - Other 1 always starts at y=0
    - Other 2 spawns at y=0 when step >= 3 and other1 has progressed

    The stochastic version is used for evaluation/training with IQ-Learn.
    """

    def __init__(self, gamma=0.99, svo_alpha=0.0):
        self.gamma = gamma
        self.svo_alpha = svo_alpha
        self.cos_alpha = np.cos(svo_alpha)
        self.sin_alpha = np.sin(svo_alpha)

        self.grid_size = GRID_SIZE
        self.num_speeds = NUM_SPEEDS
        self.actions = [SPEED_STOP, SPEED_SLOW, SPEED_FAST]

        # State dimensions:
        # ego_x: [0, grid_size-1]                    = 7
        # ego_speed: [0, 2]                           = 3
        # other1_y: [-1, grid_size-1]                 = 8 (-1=not_spawned)
        # other1_speed: [0, 2]                        = 3
        # other2_y: [-1, grid_size-1]                 = 8
        # other2_speed: [0, 2]                        = 3
        # Total: 7 * 3 * 8 * 3 * 8 * 3 = 12,096 states

        self.V = np.zeros((
            self.grid_size,   # ego_x
            self.num_speeds,  # ego_speed
            self.grid_size + 1,  # other1_y (+1 for NOT_SPAWNED mapped to index 0)
            self.num_speeds,  # other1_speed
            self.grid_size + 1,  # other2_y
            self.num_speeds,  # other2_speed
        ))

    def _y_to_idx(self, y):
        """Map y position to array index. NOT_SPAWNED (-1) → 0, y=0 → 1, etc."""
        return y + 1

    def _idx_to_y(self, idx):
        """Map array index back to y position."""
        return idx - 1

    def _simulate_step_pre_spawn(self, ego_x, ego_speed, o1_y, o1_speed, o2_y, o2_speed, action):
        """
        Simulate one step up to (but not including) respawn.
        Returns the post-move state, rewards, arrivals, and done flag.
        Respawning is handled separately by the solve loop.
        """
        # --- 1. Ego movement ---
        new_ego_speed = action
        next_ego_x = min(ego_x + new_ego_speed, self.grid_size - 1)

        # --- 2. Other vehicles' IDM-like reaction ---
        new_o1_speed, new_o2_speed = self._compute_reactions(
            ego_x, new_ego_speed, next_ego_x, o1_y, o2_y
        )

        # --- 3. Move others ---
        next_o1_y = o1_y
        if o1_y != NOT_SPAWNED and o1_y < self.grid_size - 1:
            next_o1_y = min(o1_y + new_o1_speed, self.grid_size - 1)

        next_o2_y = o2_y
        if o2_y != NOT_SPAWNED and o2_y < self.grid_size - 1:
            next_o2_y = min(o2_y + new_o2_speed, self.grid_size - 1)

        # Track arrivals BEFORE respawn (for r_global)
        o1_just_arrived = (o1_y != NOT_SPAWNED and o1_y < self.grid_size - 1
                           and next_o1_y >= self.grid_size - 1)
        o2_just_arrived = (o2_y != NOT_SPAWNED and o2_y < self.grid_size - 1
                           and next_o2_y >= self.grid_size - 1)

        # --- 4. Collision detection (pre-spawn positions) ---
        collision = False
        collided_idx = -1
        if next_ego_x == CONFLICT_X:
            if next_o1_y == CONFLICT_Y:
                collision = True
                collided_idx = 0
            elif next_o2_y == CONFLICT_Y:
                collision = True
                collided_idx = 1

        # --- 5. Rewards (computed on pre-spawn state) ---
        ego_arrived = (next_ego_x >= self.grid_size - 1)
        ego_crossed = (ego_x <= CONFLICT_X < next_ego_x)

        # r_self
        if collision:
            r_self = W_COLLISION
        else:
            r_self = W_STEP
            r_self += W_SPEED * (new_ego_speed / SPEED_FAST)
            if ego_arrived:
                r_self += W_ARRIVED

        # r_global
        if collision:
            r_global = W_OTHER_COLLISION
        else:
            r_global = 0.0
            ego_in_influence_zone = (CONFLICT_X - 1 <= ego_x <= CONFLICT_X)

            if ego_in_influence_zone:
                active_others = 0
                for oy in [o1_y, o2_y]:
                    if oy != NOT_SPAWNED and oy < self.grid_size - 1:
                        active_others += 1
                if active_others > 0:
                    r_global += R_BASELINE

                if o1_just_arrived:
                    r_global += W_OTHER_ARRIVED
                if o2_just_arrived:
                    r_global += W_OTHER_ARRIVED

                for oy, os in [(o1_y, new_o1_speed), (o2_y, new_o2_speed)]:
                    if oy == NOT_SPAWNED or oy >= self.grid_size - 1:
                        continue
                    speed_loss = SPEED_FAST - os
                    if speed_loss > 0:
                        if os == SPEED_STOP:
                            r_global += W_BLOCK
                        else:
                            r_global += W_DELAY * (speed_loss / SPEED_FAST)

            if ego_crossed:
                r_global += W_FLOW
            if ego_arrived:
                r_global += W_EGO_CLEAR

        done = ego_arrived or collision

        return (next_ego_x, new_ego_speed, next_o1_y, new_o1_speed,
                next_o2_y, new_o2_speed, o1_just_arrived, o2_just_arrived,
                r_self, r_global, done)

    def _get_spawn_outcomes(self, o1_y, o1_arrived, o2_y, o2_arrived, o1_speed, o2_speed):
        """
        Enumerate all possible respawn outcomes with probabilities.
        Each arrived vehicle independently respawns with probability SPAWN_PROB
        or stays NOT_SPAWNED with probability (1 - SPAWN_PROB).
        Not-yet-spawned other2 also has a chance to initially spawn.

        Returns: list of (probability, new_o1_y, new_o1_speed, new_o2_y, new_o2_speed)
        """
        p = SPAWN_PROB

        # Determine which vehicles need a spawn decision
        o1_needs_spawn = (o1_arrived or o1_y >= self.grid_size - 1)
        o2_needs_spawn = (o2_arrived or o2_y >= self.grid_size - 1
                          or o2_y == NOT_SPAWNED)

        # For vehicles that don't need spawn, they keep current state
        outcomes = []

        # Build o1 options
        if o1_needs_spawn:
            o1_options = [(p, 0, SPEED_FAST), (1-p, NOT_SPAWNED, SPEED_FAST)]
        else:
            o1_options = [(1.0, o1_y, o1_speed)]

        # Build o2 options
        if o2_needs_spawn:
            o2_options = [(p, 0, SPEED_FAST), (1-p, NOT_SPAWNED, SPEED_FAST)]
        else:
            o2_options = [(1.0, o2_y, o2_speed)]

        # Cross product
        for p1, y1, s1 in o1_options:
            for p2, y2, s2 in o2_options:
                prob = p1 * p2
                if prob > 1e-10:
                    outcomes.append((prob, y1, s1, y2, s2))

        return outcomes

    def _compute_reactions(self, ego_x, ego_speed, ego_next_x, o1_y, o2_y):
        """Compute IDM-like reactions for both other vehicles."""
        o1_speed = self._single_reaction(ego_x, ego_speed, ego_next_x, o1_y)
        o2_speed = self._single_reaction(ego_x, ego_speed, ego_next_x, o2_y)
        return o1_speed, o2_speed

    def _single_reaction(self, ego_x, ego_speed, ego_next_x, other_y):
        """Compute reaction for a single other vehicle."""
        if other_y == NOT_SPAWNED or other_y >= self.grid_size - 1:
            return SPEED_FAST

        desired_speed = SPEED_FAST

        # Check if ego is at or crossing conflict point
        ego_at_conflict = (
            ego_x <= CONFLICT_X <= ego_next_x
            or ego_x == CONFLICT_X
        )

        dist_to_conflict = CONFLICT_Y - other_y

        if ego_at_conflict and 0 < dist_to_conflict <= REACTION_DISTANCE:
            if dist_to_conflict == 1:
                desired_speed = SPEED_STOP
            elif dist_to_conflict <= REACTION_DISTANCE:
                desired_speed = SPEED_SLOW

        # Ego blocking the conflict point
        if ego_x == CONFLICT_X and dist_to_conflict > 0:
            if dist_to_conflict <= 2:
                desired_speed = SPEED_STOP
            else:
                desired_speed = min(desired_speed, SPEED_SLOW)

        return desired_speed

    def _compute_svo_reward(self, r_self, r_global):
        return self.cos_alpha * r_self + self.sin_alpha * r_global

    def solve(self, threshold=1e-6, max_iterations=2000):
        """Run Value Iteration with stochastic spawning (Bellman expectation)."""
        print(f"Running Value Iteration for SVO alpha = "
              f"{np.rad2deg(self.svo_alpha):.1f}° ...")
        print(f"State space size: {self.V.size}")
        print(f"Spawn probability: {SPAWN_PROB}")

        for iteration in range(max_iterations):
            delta = 0
            new_V = np.copy(self.V)

            for ex in range(self.grid_size):
                # Terminal: ego arrived
                if ex == self.grid_size - 1:
                    new_V[ex, :, :, :, :, :] = 0
                    continue

                for es in range(self.num_speeds):
                    for o1_idx in range(self.grid_size + 1):
                        o1_y = self._idx_to_y(o1_idx)
                        for o1_s in range(self.num_speeds):
                            for o2_idx in range(self.grid_size + 1):
                                o2_y = self._idx_to_y(o2_idx)
                                for o2_s in range(self.num_speeds):
                                    q_values = []
                                    for a in self.actions:
                                        # Physics step (deterministic)
                                        (nx, ns, no1y, no1s, no2y, no2s,
                                         o1_arr, o2_arr,
                                         r_self, r_global, done) = \
                                            self._simulate_step_pre_spawn(
                                                ex, es, o1_y, o1_s,
                                                o2_y, o2_s, a)

                                        r_svo = self._compute_svo_reward(
                                            r_self, r_global)

                                        if done:
                                            q_val = r_svo
                                        else:
                                            # Stochastic spawn: expected
                                            # value over respawn outcomes
                                            outcomes = self._get_spawn_outcomes(
                                                no1y, o1_arr,
                                                no2y, o2_arr,
                                                no1s, no2s)

                                            expected_v = 0.0
                                            for (prob, sy1, ss1,
                                                 sy2, ss2) in outcomes:
                                                si1 = self._y_to_idx(sy1)
                                                si2 = self._y_to_idx(sy2)
                                                expected_v += prob * \
                                                    self.V[nx, ns, si1, ss1,
                                                           si2, ss2]

                                            q_val = r_svo + self.gamma * expected_v

                                        q_values.append(q_val)

                                    best_v = max(q_values)
                                    idx = (ex, es, o1_idx, o1_s,
                                           o2_idx, o2_s)
                                    delta = max(delta,
                                                abs(best_v - self.V[idx]))
                                    new_V[idx] = best_v

            self.V = new_V
            if iteration % 50 == 0:
                print(f"  Iteration {iteration}, delta={delta:.8f}")
            if delta < threshold:
                print(f"Converged in {iteration + 1} iterations "
                      f"(delta={delta:.10f})")
                break

    def get_optimal_action(self, ego_x, ego_speed, o1_y, o1_speed, o2_y, o2_speed):
        """Returns the best action based on the converged V-table."""
        best_action = None
        best_q = -float('inf')

        for a in self.actions:
            (nx, ns, no1y, no1s, no2y, no2s,
             o1_arr, o2_arr,
             r_self, r_global, done) = \
                self._simulate_step_pre_spawn(ego_x, ego_speed, o1_y, o1_speed,
                                               o2_y, o2_speed, a)

            r_svo = self._compute_svo_reward(r_self, r_global)

            if done:
                q_val = r_svo
            else:
                outcomes = self._get_spawn_outcomes(
                    no1y, o1_arr, no2y, o2_arr, no1s, no2s)
                expected_v = 0.0
                for (prob, sy1, ss1, sy2, ss2) in outcomes:
                    si1 = self._y_to_idx(sy1)
                    si2 = self._y_to_idx(sy2)
                    expected_v += prob * self.V[nx, ns, si1, ss1, si2, ss2]
                q_val = r_svo + self.gamma * expected_v

            if q_val > best_q:
                best_q = q_val
                best_action = a

        return best_action

    def extract_trajectories(self, num_trajectories=10, seed=42):
        """Roll out the optimal policy in the stochastic env."""
        env = SVOIntersectionGridV2(deterministic_traffic=False)
        trajectories = []

        for i in range(num_trajectories):
            obs, _ = env.reset(seed=seed + i)
            trajectory = []
            done = False

            while not done:
                ex = int(obs[0])
                es = int(obs[1])
                o1y = int(obs[2])
                o1s = int(obs[3])
                o2y = int(obs[4])
                o2s = int(obs[5])

                action = self.get_optimal_action(ex, es, o1y, o1s, o2y, o2s)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                r_self = info['rewards/component_self']
                r_global = info['rewards/component_global']
                crashed = info['collision']

                transition = (
                    obs.copy(),
                    action,
                    float(r_self),
                    next_obs.copy(),
                    done,
                    crashed,
                    float(r_self),
                    float(r_global),
                )
                trajectory.append(transition)
                obs = next_obs

            trajectories.append(trajectory)

        return trajectories


# =========================================================================
# Quick test / demo
# =========================================================================

if __name__ == "__main__":
    import os
    import pickle

    print("=" * 60)
    print("SVOIntersectionGridV2 — Demo & Expert Generation")
    print("=" * 60)

    # --- 1. Quick manual rollout to verify dynamics ---
    print("\n--- Manual rollout (ego goes FAST) ---")
    env = SVOIntersectionGridV2(render_mode="console", deterministic_traffic=True)
    obs, _ = env.reset()
    env.render()
    done = False
    total_r_self = 0
    total_r_global = 0
    while not done:
        action = SPEED_FAST  # Always go fast
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_r_self += info['rewards/component_self']
        total_r_global += info['rewards/component_global']
        done = terminated or truncated
    print(f"Total r_self={total_r_self:.2f}, r_global={total_r_global:.2f}, "
          f"crashed={info['collision']}, arrived={info['ego_arrived']}")

    # --- 2. Solve with value iteration for each SVO profile ---
    svo_profiles = {
        'egoistic': 0.0,
        'prosocial': np.pi / 4,
        'altruistic': np.pi / 2,
    }

    os.makedirs('expert_demonstrations_v2', exist_ok=True)

    for name, alpha in svo_profiles.items():
        print(f"\n{'=' * 40}")
        print(f"Solving for {name} (α = {np.rad2deg(alpha):.0f}°)")
        print(f"{'=' * 40}")

        solver = SVOValueIterationV2(gamma=0.99, svo_alpha=alpha)
        solver.solve()

        # Generate demonstrations
        trajectories = solver.extract_trajectories(num_trajectories=10)

        # Save
        file_path = f'./expert_demonstrations_v2/expert_{name}.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(trajectories, f)
        print(f"Saved {len(trajectories)} {name} trajectories to {file_path}")

        # Print example trajectory
        print(f"\n--- Example {name.capitalize()} Trajectory ---")
        for step, t in enumerate(trajectories[0]):
            action_names = ["STOP", "SLOW", "FAST"]
            state = t[0]
            print(f"  Step {step}: ego_x={int(state[0])} spd={int(state[1])} | "
                  f"o1_y={int(state[2])} o2_y={int(state[4])} | "
                  f"act={action_names[t[1]]} | "
                  f"r_self={t[6]:.2f} r_global={t[7]:.2f}")
        print()