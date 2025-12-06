"""
PvZ Survival Endless Simulator - Physics Module
AABB collision detection and projectile update logic.
Optimized for high-performance simulation.
"""

from typing import Optional
import math

from consts import (
    DamageType, ProjectileType, ZombieType, ZombieState, PlantType,
    FUME_RANGE, GLOOM_RANGE, WINTER_MELON_SPLASH_RADIUS,
    CHERRY_BOMB_RADIUS, SQUASH_JUMP_RANGE,
    CELL_WIDTH, CELL_HEIGHT, GRID_OFFSET_X, GRID_OFFSET_Y,
    NUM_ROWS, ZOMBIE_SPAWN_X,
)
from state import Rect, Zombie, Plant, Projectile


def check_collision(rect1: Rect, rect2: Rect) -> bool:
    """
    Check if two rectangles collide using AABB collision detection.
    This is the core collision function - NOT grid-based.
    
    Args:
        rect1: First rectangle
        rect2: Second rectangle
        
    Returns:
        True if rectangles overlap, False otherwise
    """
    return rect1.intersects(rect2)


def check_point_in_rect(x: float, y: float, rect: Rect) -> bool:
    """Check if a point is inside a rectangle."""
    return (rect.left <= x <= rect.right and
            rect.top <= y <= rect.bottom)


def distance_squared(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate squared distance between two points (faster than sqrt)."""
    dx = x2 - x1
    dy = y2 - y1
    return dx * dx + dy * dy


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate distance between two points."""
    return math.sqrt(distance_squared(x1, y1, x2, y2))


def check_radius_collision(x1: float, y1: float, x2: float, y2: float, 
                           radius: float) -> bool:
    """Check if two points are within a radius (for splash damage)."""
    return distance_squared(x1, y1, x2, y2) <= radius * radius


def get_row_from_y(y: float) -> int:
    """Get the grid row from a Y coordinate."""
    row = int((y - GRID_OFFSET_Y) // CELL_HEIGHT)
    return max(0, min(row, NUM_ROWS - 1))


def get_col_from_x(x: float) -> int:
    """Get the grid column from an X coordinate."""
    col = int((x - GRID_OFFSET_X) // CELL_WIDTH)
    return max(0, min(col, 8))


class ProjectileManager:
    """
    Manages all projectiles in the simulation.
    Handles movement, collision detection, and damage application.
    """
    
    def __init__(self):
        self.projectiles: list[Projectile] = []
        self.next_id: int = 0
    
    def create_projectile(self, proj_type: ProjectileType,
                          x: float, y: float, row: int,
                          target_x: float = 0, target_y: float = 0,
                          source_plant_id: int = -1) -> Projectile:
        """Create and track a new projectile."""
        proj = Projectile.create(
            self.next_id, proj_type, x, y, row,
            target_x, target_y, source_plant_id
        )
        self.next_id += 1
        self.projectiles.append(proj)
        return proj
    
    def update(self, zombies: list[Zombie]) -> list[tuple[Zombie, int, DamageType]]:
        """
        Update all projectiles and check for collisions.
        
        Returns:
            List of (zombie, damage, damage_type) tuples for hits
        """
        hits: list[tuple[Zombie, int, DamageType]] = []
        to_remove: list[int] = []
        
        for i, proj in enumerate(self.projectiles):
            if not proj.mIsActive:
                to_remove.append(i)
                continue
            
            # Update position
            proj.update_position()
            
            # Check if off screen
            if proj.is_off_screen():
                proj.mIsActive = False
                to_remove.append(i)
                continue
            
            # Check for collisions with zombies
            proj_rect = proj.get_rect()
            
            for zombie in zombies:
                if not zombie.is_targetable():
                    continue
                
                # Lobbed projectiles can hit any row when landing
                if not proj.mIsLobbed and zombie.mRow != proj.mRow:
                    continue
                
                # Non-lobbed can only hit ground zombies
                if not proj.mIsLobbed and zombie.mIsFlying:
                    continue
                
                zombie_rect = zombie.get_hurt_rect()
                
                if check_collision(proj_rect, zombie_rect):
                    # Handle lobbed projectile landing
                    if proj.mIsLobbed:
                        # Check if close to target Y (landing)
                        if abs(proj.mY - proj.mTargetY) > 20:
                            continue
                    
                    # Apply damage
                    hits.append((zombie, proj.mDamage, proj.mDamageType))
                    
                    # Handle splash damage
                    if proj.mSplashRadius > 0:
                        for other_zombie in zombies:
                            if other_zombie == zombie or not other_zombie.is_targetable():
                                continue
                            if check_radius_collision(proj.mX, proj.mY,
                                                     other_zombie.mX, other_zombie.mY,
                                                     proj.mSplashRadius):
                                hits.append((other_zombie, proj.mDamage, proj.mDamageType))
                    
                    proj.mIsActive = False
                    to_remove.append(i)
                    break
        
        # Remove inactive projectiles (reverse order to preserve indices)
        for i in sorted(to_remove, reverse=True):
            self.projectiles.pop(i)
        
        return hits
    
    def clear(self) -> None:
        """Clear all projectiles."""
        self.projectiles.clear()


class CollisionSystem:
    """
    Handles all collision detection in the game.
    Uses AABB collision for zombie-plant interactions.
    """
    
    @staticmethod
    def get_zombie_plant_collisions(zombies: list[Zombie], 
                                     plants: list[Plant]) -> list[tuple[Zombie, Plant]]:
        """
        Find all zombie-plant collisions (for eating).
        Uses AABB collision between zombie AttackRect and plant HurtRect.
        
        Returns:
            List of (zombie, plant) pairs that are colliding
        """
        collisions: list[tuple[Zombie, Plant]] = []
        
        for zombie in zombies:
            if not zombie.can_eat():
                continue
            
            zombie_rect = zombie.get_attack_rect()
            
            for plant in plants:
                if not plant.is_alive():
                    continue
                
                # Must be same row
                if zombie.mRow != plant.mRow:
                    continue
                
                plant_rect = plant.get_hurt_rect()
                
                if check_collision(zombie_rect, plant_rect):
                    collisions.append((zombie, plant))
        
        return collisions
    
    @staticmethod
    def get_plants_in_range(x: float, y: float, row: int, 
                            plants: list[Plant], 
                            range_pixels: float,
                            check_row: bool = True) -> list[Plant]:
        """Get all plants within range of a point."""
        result: list[Plant] = []
        
        for plant in plants:
            if not plant.is_alive():
                continue
            if check_row and plant.mRow != row:
                continue
            
            if distance(x, y, plant.mX, plant.mY) <= range_pixels:
                result.append(plant)
        
        return result
    
    @staticmethod
    def get_zombies_in_range(x: float, y: float, row: int,
                             zombies: list[Zombie],
                             range_pixels: float,
                             check_row: bool = True,
                             ground_only: bool = False) -> list[Zombie]:
        """Get all zombies within range of a point."""
        result: list[Zombie] = []
        
        for zombie in zombies:
            if not zombie.is_targetable():
                continue
            if check_row and zombie.mRow != row:
                continue
            if ground_only and zombie.mIsFlying:
                continue
            
            if distance(x, y, zombie.mX, zombie.mY) <= range_pixels:
                result.append(zombie)
        
        return result
    
    @staticmethod
    def get_zombies_in_lane_ahead(x: float, row: int,
                                   zombies: list[Zombie],
                                   range_pixels: float) -> list[Zombie]:
        """Get zombies in lane ahead of x position within range."""
        result: list[Zombie] = []
        
        for zombie in zombies:
            if not zombie.is_targetable():
                continue
            if zombie.mRow != row:
                continue
            if zombie.mIsFlying:
                continue
            
            # Zombie must be ahead (greater X) and within range
            if x < zombie.mX <= x + range_pixels:
                result.append(zombie)
        
        return result
    
    @staticmethod
    def get_frontmost_zombie_in_lane(row: int, zombies: list[Zombie]) -> Optional[Zombie]:
        """Get the zombie closest to the house (smallest X) in a lane."""
        min_x = float('inf')
        frontmost: Optional[Zombie] = None
        
        for zombie in zombies:
            if not zombie.is_targetable():
                continue
            if zombie.mRow != row:
                continue
            
            if zombie.mX < min_x:
                min_x = zombie.mX
                frontmost = zombie
        
        return frontmost
    
    @staticmethod
    def get_plant_at(row: int, col: int, plants: list[Plant]) -> Optional[Plant]:
        """Get the plant at a specific grid position."""
        for plant in plants:
            if plant.is_alive() and plant.mRow == row and plant.mCol == col:
                return plant
        return None


class FumeAttack:
    """Handles Fume-shroom line attack logic."""
    
    @staticmethod
    def get_targets(plant: Plant, zombies: list[Zombie]) -> list[Zombie]:
        """
        Get all zombies hit by a fume-shroom attack.
        Fume attacks in a line, piercing shields.
        """
        targets: list[Zombie] = []
        
        for zombie in zombies:
            if not zombie.is_targetable():
                continue
            if zombie.mRow != plant.mRow:
                continue
            
            # Check if zombie is ahead of plant and within range
            dx = zombie.mX - plant.mX
            if 0 < dx <= FUME_RANGE:
                targets.append(zombie)
        
        return targets


class GloomAttack:
    """Handles Gloom-shroom 3x3 AOE attack logic."""
    
    @staticmethod
    def get_targets(plant: Plant, zombies: list[Zombie]) -> list[Zombie]:
        """
        Get all zombies hit by a gloom-shroom attack.
        Gloom attacks in 3x3 area around it, piercing shields.
        """
        targets: list[Zombie] = []
        adjacent_rows = [plant.mRow - 1, plant.mRow, plant.mRow + 1]
        
        for zombie in zombies:
            if not zombie.is_targetable():
                continue
            if zombie.mRow not in adjacent_rows:
                continue
            
            # Check distance from plant
            if distance(plant.mX, plant.mY, zombie.mX, zombie.mY) <= GLOOM_RANGE:
                targets.append(zombie)
        
        return targets


class WinterMelonAttack:
    """Handles Winter Melon lobbed + splash attack logic."""
    
    @staticmethod
    def calculate_trajectory(start_x: float, start_y: float,
                            target_x: float, target_y: float) -> tuple[float, float, float]:
        """
        Calculate the initial velocities and travel time for a lobbed projectile.
        Uses simplified parabolic motion.
        
        Returns:
            (vel_x, vel_y, travel_ticks)
        """
        dx = target_x - start_x
        dy = target_y - start_y
        
        # Estimate travel time based on distance
        travel_ticks = abs(dx) / 2.5  # Base speed
        
        if travel_ticks < 1:
            travel_ticks = 1
        
        vel_x = dx / travel_ticks
        
        # Calculate initial Y velocity for parabolic arc
        # Using simplified physics: y = y0 + vy*t + 0.5*g*t^2
        # Solving for vy to land at target_y
        gravity = 0.05
        vel_y = (dy - 0.5 * gravity * travel_ticks * travel_ticks) / travel_ticks
        
        return vel_x, vel_y, travel_ticks


class CherryBombAttack:
    """Handles Cherry Bomb instant 3x3 explosion."""
    
    @staticmethod
    def get_targets(plant: Plant, zombies: list[Zombie]) -> list[Zombie]:
        """Get all zombies hit by cherry bomb explosion."""
        targets: list[Zombie] = []
        
        for zombie in zombies:
            if not zombie.is_alive():
                continue
            
            # Check if zombie is within explosion radius
            if distance(plant.mX, plant.mY, zombie.mX, zombie.mY) <= CHERRY_BOMB_RADIUS:
                targets.append(zombie)
        
        return targets


class SquashAttack:
    """Handles Squash jump and smash logic."""
    
    @staticmethod
    def find_target(plant: Plant, zombies: list[Zombie]) -> Optional[Zombie]:
        """Find the closest zombie in range for squash to target."""
        closest: Optional[Zombie] = None
        min_dist = float('inf')
        
        for zombie in zombies:
            if not zombie.is_targetable():
                continue
            if zombie.mIsFlying:
                continue
            
            dist = distance(plant.mX, plant.mY, zombie.mX, zombie.mY)
            
            if dist <= SQUASH_JUMP_RANGE and dist < min_dist:
                min_dist = dist
                closest = zombie
        
        return closest


class SpikerockCollision:
    """Handles Spikerock floor damage and vehicle collision."""
    
    @staticmethod
    def get_zombies_on_tile(plant: Plant, zombies: list[Zombie]) -> list[Zombie]:
        """Get zombies standing on the spikerock tile."""
        targets: list[Zombie] = []
        
        # Get tile bounds
        tile_left = GRID_OFFSET_X + plant.mCol * CELL_WIDTH
        tile_right = tile_left + CELL_WIDTH
        
        for zombie in zombies:
            if not zombie.is_alive():
                continue
            if zombie.mRow != plant.mRow:
                continue
            if zombie.mIsFlying or zombie.mIsDigging:
                continue
            
            # Check if zombie is on the tile
            if tile_left <= zombie.mX <= tile_right:
                targets.append(zombie)
        
        return targets
    
    @staticmethod
    def check_vehicle_collision(plant: Plant, zombie: Zombie) -> bool:
        """Check if a vehicle zombie collides with spikerock."""
        if zombie.mZombieType not in (ZombieType.ZOMBONI, ZombieType.CATAPULT):
            return False
        
        if zombie.mRow != plant.mRow:
            return False
        
        # Get tile center
        tile_center_x = GRID_OFFSET_X + plant.mCol * CELL_WIDTH + CELL_WIDTH / 2
        
        # Check if vehicle is close to tile center
        return abs(zombie.mX - tile_center_x) < CELL_WIDTH / 2


class GarlicCollision:
    """Handles Garlic row redirect logic."""
    
    @staticmethod
    def redirect_zombie(zombie: Zombie, current_row: int) -> int:
        """
        Calculate the new row for a zombie that bites garlic.
        Zombies prefer to move to adjacent rows.
        """
        # Try to move away from center rows (2, 3 are pool)
        if current_row <= 0:
            return 1  # Can only go down
        if current_row >= NUM_ROWS - 1:
            return current_row - 1  # Can only go up
        
        # Prefer moving away from pool for non-pool zombies
        if current_row == 2:
            return 1  # Move up from pool
        if current_row == 3:
            return 4  # Move down from pool
        
        # Default: move up
        return current_row - 1


class LadderPlacement:
    """Handles Ladder zombie ladder placement logic."""
    
    @staticmethod
    def can_place_ladder(zombie: Zombie, plant: Plant) -> bool:
        """Check if ladder zombie can place ladder on this plant."""
        if zombie.mZombieType != ZombieType.LADDER:
            return False
        if not zombie.mHasLadder:
            return False
        if zombie.mState != ZombieState.WALK:
            return False
        
        # Can place on defensive plants (Pumpkin, Wall-nut, Tall-nut)
        # For our implementation, only Pumpkin and Garlic are relevant
        if plant.mPlantType in (PlantType.PUMPKIN, PlantType.GARLIC):
            return True
        if plant.mHasPumpkin:
            return True
        
        return False
