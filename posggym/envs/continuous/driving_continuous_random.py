"""The Driving Continuous Random Environment."""

import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import pymunk
from pymunk import Vec2d

import posggym.model as M
from posggym.envs.continuous.core import CircleEntity, Coord, FloatCoord, Position
from posggym.envs.continuous.driving_continuous import (
    DAction,
    DObs,
    DState,
    DrivingContinuousEnv,
    DrivingContinuousModel,
    DrivingWorld,
    parseworld_str,
)


class DrivingContinuousRandomEnv(DrivingContinuousEnv):
    """The Driving Continuous Random Environment.

    This environment extends the DrivingContinuous environment by adding random circular
    obstacles instead of the hard-coded block obstacles. The obstacles are generated
    randomly based on the specified parameters.

    Additional Arguments
    -------------------
    - `obstacle_radius_range` - a tuple (min_radius, max_radius) specifying the range of
        possible obstacle radii (default = `(0.3, 0.7)`).
    - `obstacle_density` - a float between 0 and 1 specifying the density of obstacles
        in the environment (default = `0.1`). Higher values mean more obstacles.
    - `random_seed` - an optional seed for the random number generator to ensure
        reproducible obstacle generation (default = `None`).

    Version History
    ---------------
    - `v0`: Initial version
    """

    def __init__(
        self,
        world: Union[str, "RandomDrivingWorld"] = "14x14Empty",
        num_agents: int = 2,
        obs_dist: float = 5.0,
        n_sensors: int = 16,
        obstacle_radius_range: Tuple[float, float] = (0.3, 0.7),
        obstacle_density: float = 0.1,
        random_seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        # Create the model first
        model = RandomDrivingContinuousModel(
            world,
            num_agents,
            obs_dist,
            n_sensors,
            obstacle_radius_range,
            obstacle_density,
            random_seed,
        )
        
        # Initialize with our custom model
        super(DrivingContinuousEnv, self).__init__(
            model,
            render_mode=render_mode,
        )
        
        # Initialize rendering attributes
        self.window_surface = None
        self.clock = None
        self.window_size = 600
        self.draw_options = None
        self.world = None
        self.blocked_surface = None


class RandomDrivingContinuousModel(DrivingContinuousModel):
    """Driving Problem Model with random obstacles.

    Parameters
    ----------
    world : str, RandomDrivingWorld
        the world environment for the model scenario
    num_agents : int
        the number of agents in the model scenario
    obs_dists : float
        number of cells in front, behind, and to the side that each agent can observe
    n_sensors : int
        the number of sensor lines eminating from the agent. The agent will observe at
        `n_sensors` equidistance intervals over `[0, 2*pi]`
    obstacle_radius_range : Tuple[float, float]
        the range of possible obstacle radii (min_radius, max_radius)
    obstacle_density : float
        a value between 0 and 1 specifying the density of obstacles
    random_seed : Optional[int]
        an optional seed for the random number generator
    """

    def __init__(
        self,
        world: Union[str, "RandomDrivingWorld"],
        num_agents: int,
        obs_dist: float,
        n_sensors: int,
        obstacle_radius_range: Tuple[float, float],
        obstacle_density: float,
        random_seed: Optional[int] = None,
    ):
        if isinstance(world, str):
            # Create a RandomDrivingWorld instead of a regular DrivingWorld
            if world in SUPPORTED_RANDOM_WORLDS:
                world_info = SUPPORTED_RANDOM_WORLDS[world]
                world = parse_random_world_str(
                    world_info["world_str"],
                    world_info["supported_num_agents"],
                    obstacle_radius_range,
                    obstacle_density,
                    random_seed,
                )
            else:
                # Try to use the standard worlds but with random obstacles
                assert world in STANDARD_WORLDS, (
                    f"Unsupported world '{world}'. If world argument is a string it must "
                    f"be one of: {list(SUPPORTED_RANDOM_WORLDS.keys()) + list(STANDARD_WORLDS.keys())}."
                )
                world_info = STANDARD_WORLDS[world]
                world = parse_random_world_str(
                    world_info["world_str"],
                    world_info["supported_num_agents"],
                    obstacle_radius_range,
                    obstacle_density,
                    random_seed,
                )

        # Initialize the parent class with our RandomDrivingWorld
        super().__init__(world, num_agents, obs_dist, n_sensors)


class RandomDrivingWorld(DrivingWorld):
    """A world for the Driving Problem with random obstacles."""

    def __init__(
        self,
        size: int,
        blocked_coords: Set[Coord],
        start_coords: List[Set[FloatCoord]],
        dest_coords: List[Set[FloatCoord]],
        obstacle_radius_range: Tuple[float, float] = (0.3, 0.7),
        obstacle_density: float = 0.1,
        random_seed: Optional[int] = None,
    ):
        # Initialize with empty blocked_coords since we'll add random obstacles
        super().__init__(size, blocked_coords, start_coords, dest_coords)
        
        # Generate random obstacles
        self.blocks = self._generate_random_obstacles(
            size, obstacle_radius_range, obstacle_density, random_seed
        )
        
        # Add the obstacles to the physics space
        for pos, radius in self.blocks:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Circle(body, radius)
            body.position = Vec2d(pos[0], pos[1])
            shape.elasticity = 0.0  # no bouncing
            shape.color = self.BLOCK_COLOR
            shape.collision_type = self.get_collision_id()
            self.space.add(body, shape)
        
        # Reset blocked_coords to be recalculated based on the new obstacles
        # self._blocked_coords = None

    def _generate_random_obstacles(
        self,
        size: int,
        obstacle_radius_range: Tuple[float, float],
        obstacle_density: float,
        random_seed: Optional[int] = None,
    ) -> List[CircleEntity]:
        """Generate random circular obstacles.
        
        Parameters
        ----------
        size : int
            The size of the world
        obstacle_radius_range : Tuple[float, float]
            The range of possible obstacle radii (min_radius, max_radius)
        obstacle_density : float
            A value between 0 and 1 specifying the density of obstacles
        random_seed : Optional[int]
            An optional seed for the random number generator
            
        Returns
        -------
        List[CircleEntity]
            A list of circular obstacles, each defined by a position and radius
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Validate parameters
        min_radius, max_radius = obstacle_radius_range
        assert 0 < min_radius <= max_radius, "Invalid obstacle radius range"
        assert 0 <= obstacle_density <= 1, "Obstacle density must be between 0 and 1"
        
        # Calculate the number of obstacles based on density
        # Higher density means more obstacles
        area = size * size
        avg_obstacle_area = math.pi * ((min_radius + max_radius) / 2) ** 2
        max_obstacles = int(area / avg_obstacle_area * 0.5)  # Limit to 50% coverage
        num_obstacles = int(max_obstacles * obstacle_density)
        
        obstacles = []
        
        # Keep track of start and destination areas to avoid placing obstacles there
        protected_areas = []
        for starts in self.start_coords:
            for x, y in starts:
                protected_areas.append((x, y, self.agent_radius * 2))
        
        for dests in self.dest_coords:
            for x, y in dests:
                protected_areas.append((x, y, self.agent_radius * 2))
        
        # Generate obstacles
        attempts = 0
        max_attempts = num_obstacles * 10  # Limit the number of attempts
        
        while len(obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1
            
            # Generate random position and radius
            radius = random.uniform(min_radius, max_radius)
            x = random.uniform(radius, size - radius)
            y = random.uniform(radius, size - radius)
            
            # Check if the obstacle overlaps with protected areas
            overlap = False
            for px, py, pr in protected_areas:
                if math.sqrt((x - px) ** 2 + (y - py) ** 2) < (radius + pr):
                    overlap = True
                    break
            
            # Check if the obstacle overlaps with existing obstacles
            for pos, r in obstacles:
                if math.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2) < (radius + r + 0.5):
                    overlap = True
                    break
            
            if not overlap:
                obstacles.append(((x, y, 0), radius))  # Position includes angle=0
        
        return obstacles

    def copy(self) -> "RandomDrivingWorld":
        """Get a deep copy of this world."""
        assert self._blocked_coords is not None
        world = RandomDrivingWorld(
            size=int(self.size),
            blocked_coords=self._blocked_coords,
            start_coords=self.start_coords,
            dest_coords=self.dest_coords,
        )
        
        # Copy the blocks directly instead of regenerating them
        world.blocks = self.blocks.copy()
        
        # Add the blocks to the physics space
        for pos, radius in world.blocks:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Circle(body, radius)
            body.position = Vec2d(pos[0], pos[1])
            shape.elasticity = 0.0  # no bouncing
            shape.color = self.BLOCK_COLOR
            shape.collision_type = world.get_collision_id()
            world.space.add(body, shape)
        
        # Copy entities
        for id, (body, shape) in self.entities.items():
            # make copies of each entity, and ensure the copies are linked correctly
            # and added to the new world and world space
            body = body.copy()
            shape = shape.copy()
            shape.body = body
            world.space.add(body, shape)
            world.entities[id] = (body, shape)
        
        return world


def parse_random_world_str(
    world_str: str,
    supported_num_agents: int,
    obstacle_radius_range: Tuple[float, float],
    obstacle_density: float,
    random_seed: Optional[int] = None,
) -> RandomDrivingWorld:
    """Parse a str representation of a world and add random obstacles.
    
    This function is similar to parseworld_str but creates a RandomDrivingWorld
    with random obstacles instead of a regular DrivingWorld.
    """
    # Parse the world string to get the basic structure
    lines = world_str.strip().split("\n")
    height = len(lines)
    width = max(len(line) for line in lines)
    
    # Ensure all lines have the same length
    lines = [line.ljust(width) for line in lines]
    
    blocked_coords: Set[Coord] = set()
    
    # Lists of sets of coordinates
    # Each list index corresponds to an agent index
    # Each set contains the possible starting/destination coordinates for that agent
    start_coords: List[Set[FloatCoord]] = [set() for _ in range(10)]
    dest_coords: List[Set[FloatCoord]] = [set() for _ in range(10)]
    
    # Any agent can start at these coordinates
    any_agent_start_coords: Set[FloatCoord] = set()
    
    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == "#":
                blocked_coords.add((c, r))
            elif char in "0123456789":
                # Starting location for specific agent
                agent_idx = int(char)
                start_coords[agent_idx].add((c + 0.5, r + 0.5))
            elif char == "+":
                # Starting location for any agent
                any_agent_start_coords.add((c + 0.5, r + 0.5))
            elif "a" <= char <= "j":
                # Destination location for specific agent
                agent_idx = ord(char) - ord("a")
                dest_coords[agent_idx].add((c + 0.5, r + 0.5))
    
    # Add any_agent_start_coords to each agent's start_coords
    for agent_idx in range(10):
        start_coords[agent_idx].update(any_agent_start_coords)
    
    # Create the RandomDrivingWorld with random obstacles
    return RandomDrivingWorld(
        size=width,
        blocked_coords=blocked_coords,
        start_coords=start_coords,
        dest_coords=dest_coords,
        obstacle_radius_range=obstacle_radius_range,
        obstacle_density=obstacle_density,
        random_seed=random_seed,
    )


# Import at the end to avoid circular imports
from posggym.envs.continuous.core import PMBodyState
from posggym.envs.continuous.driving_continuous import SUPPORTED_WORLDS as STANDARD_WORLDS


# Define supported worlds for the random environment
SUPPORTED_RANDOM_WORLDS: Dict[str, Dict[str, Any]] = {
    "14x14Empty": {
        "world_str": "\n".join(["." * 14] * 14),
        "supported_num_agents": 4,
        "max_episode_steps": 100,
    },
    # Add more custom worlds here if needed
}

# Add agent start and destination positions to the empty world
for r in range(14):
    for c in range(14):
        if (r == 0 or r == 13) and (c == 0 or c == 13):
            # Add agent start positions at the corners
            row_list = list(SUPPORTED_RANDOM_WORLDS["14x14Empty"]["world_str"].split("\n"))
            row = list(row_list[r])
            row[c] = "+"
            row_list[r] = "".join(row)
            SUPPORTED_RANDOM_WORLDS["14x14Empty"]["world_str"] = "\n".join(row_list)

# Add destination positions in the middle
middle_positions = [(3, 3), (3, 10), (10, 3), (10, 10)]
for i, (r, c) in enumerate(middle_positions):
    if i < 4:  # Only add destinations for up to 4 agents
        row_list = list(SUPPORTED_RANDOM_WORLDS["14x14Empty"]["world_str"].split("\n"))
        row = list(row_list[r])
        row[c] = chr(ord('a') + i)  # a, b, c, d
        row_list[r] = "".join(row)
        SUPPORTED_RANDOM_WORLDS["14x14Empty"]["world_str"] = "\n".join(row_list)