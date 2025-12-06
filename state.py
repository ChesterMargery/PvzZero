"""
PvZ Survival Endless Simulator - State Module
Dataclasses for Plant, Zombie, and Projectile entities.
Optimized for high-performance simulation.
"""

from dataclasses import dataclass, field
from typing import Optional
import math

from consts import (
    ZombieType, ZombieState, PlantType, ProjectileType, DamageType,
    ZOMBIE_STATS, ZOMBIE_ATTACK_RECT, ZOMBIE_HURT_RECT,
    PLANT_STATS, PLANT_HURT_RECT, PROJECTILE_STATS,
    CHILL_DURATION, FREEZE_DURATION, BUTTER_DURATION,
    CHILL_SPEED_MULT, CHILL_ATTACK_MULT,
    CELL_WIDTH, CELL_HEIGHT, GRID_OFFSET_X, GRID_OFFSET_Y,
)


@dataclass(slots=True)
class Rect:
    """Axis-Aligned Bounding Box for collision detection."""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def left(self) -> float:
        return self.x
    
    @property
    def right(self) -> float:
        return self.x + self.width
    
    @property
    def top(self) -> float:
        return self.y
    
    @property
    def bottom(self) -> float:
        return self.y + self.height
    
    def intersects(self, other: 'Rect') -> bool:
        """Check if this rect intersects with another rect (AABB collision)."""
        return (self.left < other.right and
                self.right > other.left and
                self.top < other.bottom and
                self.bottom > other.top)


@dataclass(slots=True)
class StatusEffect:
    """Status effect applied to a zombie."""
    chill_timer: int = 0            # Ticks remaining for slow effect
    freeze_timer: int = 0           # Ticks remaining for freeze effect
    butter_timer: int = 0           # Ticks remaining for butter stun
    
    @property
    def is_chilled(self) -> bool:
        return self.chill_timer > 0
    
    @property
    def is_frozen(self) -> bool:
        return self.freeze_timer > 0
    
    @property
    def is_buttered(self) -> bool:
        return self.butter_timer > 0
    
    @property
    def is_immobilized(self) -> bool:
        return self.is_frozen or self.is_buttered
    
    def get_speed_mult(self) -> float:
        """Get speed multiplier based on status effects."""
        if self.is_immobilized:
            return 0.0
        if self.is_chilled:
            return CHILL_SPEED_MULT
        return 1.0
    
    def get_attack_mult(self) -> float:
        """Get attack rate multiplier based on status effects."""
        if self.is_immobilized:
            return 0.0
        if self.is_chilled:
            return CHILL_ATTACK_MULT
        return 1.0
    
    def apply_chill(self) -> None:
        """Apply chill (slow) effect."""
        self.chill_timer = CHILL_DURATION
    
    def apply_freeze(self, duration: int = FREEZE_DURATION) -> None:
        """Apply freeze (stop) effect."""
        self.freeze_timer = duration
        # Freeze removes chill
        self.chill_timer = 0
    
    def apply_butter(self) -> None:
        """Apply butter (stun) effect."""
        self.butter_timer = BUTTER_DURATION
    
    def tick(self) -> None:
        """Update status effect timers."""
        if self.chill_timer > 0:
            self.chill_timer -= 1
        if self.freeze_timer > 0:
            self.freeze_timer -= 1
        if self.butter_timer > 0:
            self.butter_timer -= 1


@dataclass(slots=True)
class Zombie:
    """
    Zombie entity with full state machine.
    Coordinates are in pixel space.
    """
    # Identification
    mZombieId: int
    mZombieType: ZombieType
    
    # Position (pixel coordinates)
    mX: float
    mY: float
    mRow: int
    
    # Health
    mHP: int
    mAccessoryHP: int               # Shield/Cone/Bucket HP
    mMaxHP: int
    mMaxAccessoryHP: int
    
    # State machine
    mState: ZombieState = ZombieState.WALK
    mStateTimer: int = 0            # Ticks in current state
    
    # Combat
    mSpeed: float = 0.0             # Pixels per tick (base)
    mAttackDamage: int = 100
    mAttackInterval: int = 40       # Ticks between attacks
    mAttackTimer: int = 0           # Ticks until next attack
    
    # Status effects
    mStatus: StatusEffect = field(default_factory=StatusEffect)
    
    # Special state flags
    mHasJumped: bool = False        # Pole Vaulter, Dolphin
    mHasLadder: bool = True         # Ladder zombie has ladder
    mHasThrownImp: bool = False     # Gargantuar threw imp
    mIsSubmerged: bool = False      # Snorkel underwater
    mIsFlying: bool = False         # Balloon in air
    mIsDigging: bool = False        # Miner underground
    mTargetPlantId: int = -1        # Current target plant
    mSummonTimer: int = 0           # Dancer summon cooldown
    
    # Hitbox cache (updated when position changes)
    _attack_rect: Optional[Rect] = field(default=None, repr=False)
    _hurt_rect: Optional[Rect] = field(default=None, repr=False)
    
    @classmethod
    def create(cls, zombie_id: int, zombie_type: ZombieType, 
               row: int, x: float, y: float) -> 'Zombie':
        """Factory method to create a zombie with proper stats."""
        stats = ZOMBIE_STATS[zombie_type]
        base_hp, accessory_hp, speed, attack_dmg, attack_interval = stats
        
        zombie = cls(
            mZombieId=zombie_id,
            mZombieType=zombie_type,
            mX=x,
            mY=y,
            mRow=row,
            mHP=base_hp,
            mAccessoryHP=accessory_hp,
            mMaxHP=base_hp,
            mMaxAccessoryHP=accessory_hp,
            mSpeed=speed,
            mAttackDamage=attack_dmg,
            mAttackInterval=attack_interval,
        )
        
        # Set initial state based on type
        if zombie_type == ZombieType.BALLOON:
            zombie.mState = ZombieState.FLYING
            zombie.mIsFlying = True
        elif zombie_type == ZombieType.SNORKEL:
            zombie.mState = ZombieState.SUBMERGED
            zombie.mIsSubmerged = True
        elif zombie_type == ZombieType.MINER:
            zombie.mState = ZombieState.SPECIAL
            zombie.mIsDigging = True
        elif zombie_type in (ZombieType.ZOMBONI, ZombieType.CATAPULT):
            zombie.mState = ZombieState.DRIVING
        elif zombie_type == ZombieType.BUNGEE:
            zombie.mState = ZombieState.DESCENDING
        
        return zombie
    
    def get_attack_rect(self) -> Rect:
        """Get the attack collision rectangle."""
        if self._attack_rect is None:
            offsets = ZOMBIE_ATTACK_RECT[self.mZombieType]
            self._attack_rect = Rect(
                self.mX + offsets[0],
                self.mY + offsets[1],
                offsets[2],
                offsets[3]
            )
        return self._attack_rect
    
    def get_hurt_rect(self) -> Rect:
        """Get the hurt/collision rectangle."""
        if self._hurt_rect is None:
            offsets = ZOMBIE_HURT_RECT[self.mZombieType]
            self._hurt_rect = Rect(
                self.mX + offsets[0],
                self.mY + offsets[1],
                offsets[2],
                offsets[3]
            )
        return self._hurt_rect
    
    def invalidate_rects(self) -> None:
        """Invalidate cached rectangles (call when position changes)."""
        self._attack_rect = None
        self._hurt_rect = None
    
    def move(self, dx: float, dy: float = 0) -> None:
        """Move the zombie by delta values."""
        self.mX += dx
        self.mY += dy
        self.invalidate_rects()
    
    def set_position(self, x: float, y: float) -> None:
        """Set absolute position."""
        self.mX = x
        self.mY = y
        self.invalidate_rects()
    
    def get_effective_speed(self) -> float:
        """Get speed accounting for status effects."""
        return self.mSpeed * self.mStatus.get_speed_mult()
    
    def get_effective_attack_interval(self) -> int:
        """Get attack interval accounting for status effects."""
        mult = self.mStatus.get_attack_mult()
        if mult == 0:
            return 999999  # Effectively infinite
        return int(self.mAttackInterval / mult)
    
    def take_damage(self, damage: int, damage_type: DamageType = DamageType.NORMAL) -> bool:
        """
        Apply damage to zombie. Returns True if zombie died.
        Pierce damage ignores accessory (shield).
        """
        # Apply status effect for freeze damage
        if damage_type == DamageType.FREEZE:
            self.mStatus.apply_chill()
        
        # Pierce damage bypasses accessory
        if damage_type == DamageType.PIERCE or self.mAccessoryHP <= 0:
            self.mHP -= damage
        else:
            # Damage goes to accessory first
            if damage <= self.mAccessoryHP:
                self.mAccessoryHP -= damage
            else:
                remaining = damage - self.mAccessoryHP
                self.mAccessoryHP = 0
                self.mHP -= remaining
        
        if self.mHP <= 0:
            self.mState = ZombieState.DYING
            return True
        return False
    
    def is_alive(self) -> bool:
        """Check if zombie is still alive."""
        return self.mState not in (ZombieState.DYING, ZombieState.DEAD)
    
    def is_targetable(self) -> bool:
        """Check if zombie can be targeted by projectiles."""
        if not self.is_alive():
            return False
        if self.mIsSubmerged:
            return False  # Snorkel underwater
        if self.mIsDigging:
            return False  # Miner underground
        return True
    
    def is_ground_targetable(self) -> bool:
        """Check if zombie can be hit by ground attacks (not flying)."""
        return self.is_targetable() and not self.mIsFlying
    
    def can_eat(self) -> bool:
        """Check if zombie is in a state where it can eat."""
        return (self.is_alive() and 
                not self.mStatus.is_immobilized and
                self.mState in (ZombieState.WALK, ZombieState.EAT))
    
    @property
    def total_hp(self) -> int:
        """Get total HP including accessory."""
        return self.mHP + self.mAccessoryHP


@dataclass(slots=True)
class Plant:
    """
    Plant entity.
    Position is stored as grid coordinates (row, col) and pixel coordinates.
    """
    # Identification
    mPlantId: int
    mPlantType: PlantType
    
    # Grid position
    mRow: int
    mCol: int
    
    # Pixel position (center of plant)
    mX: float
    mY: float
    
    # Health
    mHP: int
    mMaxHP: int
    
    # Combat
    mAttackDamage: int = 0
    mAttackInterval: int = 0        # Ticks between attacks
    mAttackTimer: int = 0           # Ticks until next attack
    
    # State
    mIsActive: bool = True          # False if destroyed
    mIsAwake: bool = True           # Mushrooms need coffee
    mStateTimer: int = 0            # General purpose timer
    
    # Pumpkin support (can have plant inside)
    mPumpkinHP: int = 0             # HP of pumpkin layer
    mHasPumpkin: bool = False
    
    # Squash state
    mSquashTargetId: int = -1
    mSquashJumping: bool = False
    
    # Sun production
    mSunTimer: int = 0              # Ticks until next sun
    
    # Spikerock durability
    mVehicleHits: int = 0           # Hits taken from vehicles
    
    # Hitbox cache
    _hurt_rect: Optional[Rect] = field(default=None, repr=False)
    
    @classmethod
    def create(cls, plant_id: int, plant_type: PlantType, 
               row: int, col: int) -> 'Plant':
        """Factory method to create a plant with proper stats."""
        stats = PLANT_STATS[plant_type]
        hp, cost, recharge, attack_dmg, attack_interval = stats
        
        # Calculate pixel position (center of cell)
        x = GRID_OFFSET_X + col * CELL_WIDTH + CELL_WIDTH / 2
        y = GRID_OFFSET_Y + row * CELL_HEIGHT + CELL_HEIGHT / 2
        
        plant = cls(
            mPlantId=plant_id,
            mPlantType=plant_type,
            mRow=row,
            mCol=col,
            mX=x,
            mY=y,
            mHP=hp,
            mMaxHP=hp,
            mAttackDamage=attack_dmg,
            mAttackInterval=attack_interval,
        )
        
        # Mushrooms start asleep (need coffee bean)
        if plant_type in (PlantType.FUME_SHROOM, PlantType.GLOOM_SHROOM):
            plant.mIsAwake = False
        
        # Set initial sun timer
        if plant_type in (PlantType.SUNFLOWER, PlantType.TWIN_SUNFLOWER):
            plant.mSunTimer = plant.mAttackInterval
        
        return plant
    
    def get_hurt_rect(self) -> Rect:
        """Get the hurt/collision rectangle."""
        if self._hurt_rect is None:
            offsets = PLANT_HURT_RECT.get(self.mPlantType, (0, 0, 80, 80))
            cell_x = GRID_OFFSET_X + self.mCol * CELL_WIDTH
            cell_y = GRID_OFFSET_Y + self.mRow * CELL_HEIGHT
            self._hurt_rect = Rect(
                cell_x + offsets[0],
                cell_y + offsets[1],
                offsets[2],
                offsets[3]
            )
        return self._hurt_rect
    
    def take_damage(self, damage: int) -> bool:
        """
        Apply damage to plant. Returns True if plant died.
        Damage goes to pumpkin first if present.
        """
        if self.mHasPumpkin and self.mPumpkinHP > 0:
            if damage <= self.mPumpkinHP:
                self.mPumpkinHP -= damage
                return False
            else:
                remaining = damage - self.mPumpkinHP
                self.mPumpkinHP = 0
                self.mHasPumpkin = False
                self.mHP -= remaining
        else:
            self.mHP -= damage
        
        if self.mHP <= 0:
            self.mIsActive = False
            return True
        return False
    
    def add_pumpkin(self) -> None:
        """Add a pumpkin layer to this plant."""
        pumpkin_hp = PLANT_STATS[PlantType.PUMPKIN][0]
        self.mPumpkinHP = pumpkin_hp
        self.mHasPumpkin = True
        # Update hurt rect to pumpkin size
        self._hurt_rect = None
    
    def wake_up(self) -> None:
        """Wake up a sleeping mushroom (coffee bean effect)."""
        self.mIsAwake = True
    
    def is_alive(self) -> bool:
        """Check if plant is still alive."""
        return self.mIsActive and self.mHP > 0
    
    def can_attack(self) -> bool:
        """Check if plant can attack this tick."""
        return (self.is_alive() and 
                self.mIsAwake and 
                self.mAttackInterval > 0 and
                self.mAttackTimer <= 0)
    
    @property
    def total_hp(self) -> int:
        """Get total HP including pumpkin."""
        return self.mHP + self.mPumpkinHP


@dataclass(slots=True)
class Projectile:
    """
    Projectile entity (peas, spores, winter melons, etc.)
    """
    # Identification
    mProjectileId: int
    mProjectileType: ProjectileType
    
    # Position
    mX: float
    mY: float
    mRow: int
    
    # Movement
    mVelX: float = 0.0              # Horizontal velocity
    mVelY: float = 0.0              # Vertical velocity (for lobbed)
    mTargetX: float = 0.0           # For lobbed projectiles
    mTargetY: float = 0.0
    
    # Properties
    mDamage: int = 20
    mDamageType: DamageType = DamageType.NORMAL
    mSplashRadius: float = 0.0
    mIsLobbed: bool = False
    mIsActive: bool = True
    
    # Source info
    mSourcePlantId: int = -1
    
    @classmethod
    def create(cls, proj_id: int, proj_type: ProjectileType,
               x: float, y: float, row: int, 
               target_x: float = 0, target_y: float = 0,
               source_plant_id: int = -1) -> 'Projectile':
        """Factory method to create a projectile."""
        stats = PROJECTILE_STATS[proj_type]
        damage, speed, is_lobbed, splash, dmg_type = stats
        
        proj = cls(
            mProjectileId=proj_id,
            mProjectileType=proj_type,
            mX=x,
            mY=y,
            mRow=row,
            mDamage=damage,
            mDamageType=dmg_type,
            mSplashRadius=splash,
            mIsLobbed=is_lobbed,
            mSourcePlantId=source_plant_id,
        )
        
        if is_lobbed:
            # Calculate lobbed trajectory
            proj.mTargetX = target_x
            proj.mTargetY = target_y
            # Simplified parabolic motion
            dx = target_x - x
            # Estimate time to reach target
            travel_time = abs(dx) / (speed + 0.01)
            proj.mVelX = dx / max(travel_time, 1)
            # Initial upward velocity for arc
            proj.mVelY = -abs(dx) * 0.01  # Negative is up
        else:
            proj.mVelX = speed
            proj.mVelY = 0
        
        return proj
    
    def get_rect(self) -> Rect:
        """Get the projectile collision rectangle."""
        # Projectiles have small hitboxes
        return Rect(self.mX - 10, self.mY - 10, 20, 20)
    
    def update_position(self) -> None:
        """Update projectile position based on velocity."""
        self.mX += self.mVelX
        
        if self.mIsLobbed:
            self.mY += self.mVelY
            # Gravity effect
            self.mVelY += 0.05  # Gravity constant
            
            # Check if reached target (below target Y)
            if self.mY >= self.mTargetY and self.mVelY > 0:
                self.mY = self.mTargetY
    
    def is_off_screen(self) -> bool:
        """Check if projectile is off screen."""
        return self.mX < -50 or self.mX > 850 or self.mY > 650


@dataclass(slots=True)
class Sun:
    """Sun entity for economy."""
    mSunId: int
    mX: float
    mY: float
    mValue: int = 25
    mTimer: int = 1000              # Ticks before disappearing
    mIsCollected: bool = False


@dataclass(slots=True) 
class LawnMower:
    """Lawn mower entity for last-line defense."""
    mRow: int
    mX: float
    mIsActive: bool = True
    mIsTriggered: bool = False
    mSpeed: float = 3.0             # Pixels per tick when triggered
