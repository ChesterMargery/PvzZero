"""
PvZ Survival Endless Simulator - Constants Module
All values extracted from AvZ (AsmVsZombies) and RebuildPvZ reverse-engineering projects.
Tick rate: 100 ticks = 1 second
Coordinate System: X (0-800), Y (0-600)
"""

from enum import IntEnum, auto

# =============================================================================
# TIMING CONSTANTS
# =============================================================================
TICKS_PER_SECOND = 100
FRAME_DURATION = 1.0 / TICKS_PER_SECOND  # 0.01 seconds per tick

# =============================================================================
# COORDINATE SYSTEM
# =============================================================================
GAME_WIDTH = 800
GAME_HEIGHT = 600

# Grid dimensions
NUM_ROWS = 6  # Pool map has 6 rows (0-5)
NUM_COLS = 9  # 9 columns (0-8)

# Grid cell dimensions
CELL_WIDTH = 80
CELL_HEIGHT = 100

# Grid offset from game origin
GRID_OFFSET_X = 40
GRID_OFFSET_Y = 80

# Pool rows (for water detection)
POOL_ROWS = (2, 3)

# Zombie spawn X position
ZOMBIE_SPAWN_X = 800


# =============================================================================
# DAMAGE TYPES
# =============================================================================
class DamageType(IntEnum):
    NORMAL = 0
    FREEZE = 1      # Blue damage (Snow Pea, Winter Melon)
    EXPLOSION = 2   # Ash damage (Cherry Bomb, etc.)
    PIERCE = 3      # Fume-shroom, Gloom-shroom - ignores shield


# =============================================================================
# STATUS EFFECT DURATIONS (in ticks)
# =============================================================================
CHILL_DURATION = 1000           # Slow effect duration
FREEZE_DURATION = 400           # Freeze effect base duration
FREEZE_DURATION_MAX = 600       # Freeze effect max duration
BUTTER_DURATION = 400           # Butter stun duration

# Status effect multipliers
CHILL_SPEED_MULT = 0.5          # Speed multiplied by this when chilled
CHILL_ATTACK_MULT = 0.5         # Attack rate multiplied by this when chilled


# =============================================================================
# ZOMBIE TYPES
# =============================================================================
class ZombieType(IntEnum):
    ZOMBIE = 0
    FLAG = 1
    CONEHEAD = 2
    BUCKETHEAD = 3
    SCREEN_DOOR = 4
    LADDER = 5
    POLE_VAULTER = 6
    FOOTBALL = 7
    ZOMBONI = 8
    CATAPULT = 9
    SNORKEL = 10
    DOLPHIN = 11
    BALLOON = 12
    MINER = 13
    DANCER = 14
    BACKUP_DANCER = 15
    JACK_IN_THE_BOX = 16
    BUNGEE = 17
    GARGANTUAR = 18
    GIGA_GARGANTUAR = 19
    IMP = 20


# =============================================================================
# ZOMBIE STATES
# =============================================================================
class ZombieState(IntEnum):
    WALK = 0
    EAT = 1
    SPECIAL = 2         # Vaulting, Digging, Driving, etc.
    DIE = 3
    DYING = 4           # Death animation
    DEAD = 5            # Remove from game
    SUBMERGED = 6       # Snorkel underwater
    FLYING = 7          # Balloon
    RISING = 8          # Miner rising
    DRIVING = 9         # Zomboni, Catapult
    JUMPING = 10        # Pole Vaulter, Dolphin
    SMASHING = 11       # Gargantuar smash
    THROWING_IMP = 12   # Gargantuar throwing imp
    PLACING_LADDER = 13 # Ladder zombie placing ladder
    DANCING = 14        # Dancer summoning
    EXPLODING = 15      # Jack-in-the-box
    DESCENDING = 16     # Bungee descending
    GRABBING = 17       # Bungee grabbing plant
    ASCENDING = 18      # Bungee ascending


# =============================================================================
# ZOMBIE STATS (HP, Speed, Attack values from AvZ)
# Format: (base_hp, accessory_hp, speed, attack_damage, attack_interval)
# Speed is in pixels per tick
# Attack interval in ticks
# =============================================================================
ZOMBIE_STATS = {
    # (base_hp, accessory_hp, speed_per_tick, attack_dmg, attack_interval_ticks)
    ZombieType.ZOMBIE: (270, 0, 0.23, 100, 40),
    ZombieType.FLAG: (270, 0, 0.36, 100, 40),
    ZombieType.CONEHEAD: (270, 370, 0.23, 100, 40),     # 370 cone HP
    ZombieType.BUCKETHEAD: (270, 1100, 0.23, 100, 40),  # 1100 bucket HP
    ZombieType.SCREEN_DOOR: (270, 1100, 0.23, 100, 40), # 1100 screen door HP
    ZombieType.LADDER: (270, 500, 0.23, 100, 40),       # 500 ladder HP
    ZombieType.POLE_VAULTER: (335, 0, 0.70, 100, 40),   # Fast runner
    ZombieType.FOOTBALL: (1670, 0, 0.46, 100, 40),      # High HP, fast
    ZombieType.ZOMBONI: (1350, 0, 0.22, 9999, 1),       # Crushes instantly
    ZombieType.CATAPULT: (850, 0, 0.22, 75, 150),       # Basketball damage
    ZombieType.SNORKEL: (270, 0, 0.23, 100, 40),
    ZombieType.DOLPHIN: (270, 0, 0.60, 100, 40),        # Fast with dolphin
    ZombieType.BALLOON: (270, 0, 0.15, 100, 40),        # Slow flyer
    ZombieType.MINER: (270, 0, 0.23, 100, 40),
    ZombieType.DANCER: (500, 0, 0.23, 100, 40),
    ZombieType.BACKUP_DANCER: (270, 0, 0.23, 100, 40),
    ZombieType.JACK_IN_THE_BOX: (500, 0, 0.46, 9999, 1),  # Insta-kill on explode
    ZombieType.BUNGEE: (450, 0, 0.0, 9999, 1),            # Steals plant
    ZombieType.GARGANTUAR: (3000, 0, 0.18, 9999, 1),      # Smash insta-kill
    ZombieType.GIGA_GARGANTUAR: (6000, 0, 0.18, 9999, 1), # Double HP Garg
    ZombieType.IMP: (270, 0, 0.46, 100, 40),              # Fast, thrown by Garg
}

# Special zombie timings (in ticks)
POLE_VAULT_JUMP_DURATION = 60
DOLPHIN_JUMP_DURATION = 50
MINER_DIG_TIME = 600                # Time to dig across
MINER_RISE_STUN = 100               # Stun after rising
DANCER_SUMMON_CD = 450              # Cooldown between summons
JACK_EXPLODE_CHANCE = 0.003         # Per tick chance to explode
GARG_SMASH_WINDUP = 30              # Ticks before smash hits
GARG_SMASH_COOLDOWN = 120           # Ticks between smashes
GARG_THROW_IMP_HP_THRESHOLD = 0.5   # Throw imp at 50% HP
BUNGEE_DESCENT_TIME = 300           # Ticks to descend
BUNGEE_GRAB_TIME = 100              # Time to grab plant
BUNGEE_ASCENT_TIME = 200            # Ticks to ascend


# =============================================================================
# ZOMBIE HITBOXES (from AvZ)
# AttackRect: (x_offset, y_offset, width, height) relative to zombie position
# These determine when zombie can attack plants
# =============================================================================
ZOMBIE_ATTACK_RECT = {
    ZombieType.ZOMBIE: (-10, 0, 50, 100),
    ZombieType.FLAG: (-10, 0, 50, 100),
    ZombieType.CONEHEAD: (-10, 0, 50, 100),
    ZombieType.BUCKETHEAD: (-10, 0, 50, 100),
    ZombieType.SCREEN_DOOR: (-10, 0, 50, 100),
    ZombieType.LADDER: (-10, 0, 50, 100),
    ZombieType.POLE_VAULTER: (-10, 0, 50, 100),
    ZombieType.FOOTBALL: (-10, 0, 50, 100),
    ZombieType.ZOMBONI: (-20, 0, 120, 100),      # Wider for vehicle
    ZombieType.CATAPULT: (-20, 0, 120, 100),
    ZombieType.SNORKEL: (-10, 0, 50, 100),
    ZombieType.DOLPHIN: (-10, 0, 50, 100),
    ZombieType.BALLOON: (-10, 0, 50, 100),
    ZombieType.MINER: (-10, 0, 50, 100),
    ZombieType.DANCER: (-10, 0, 50, 100),
    ZombieType.BACKUP_DANCER: (-10, 0, 50, 100),
    ZombieType.JACK_IN_THE_BOX: (-10, 0, 50, 100),
    ZombieType.BUNGEE: (0, 0, 60, 100),          # Bungee has different rect
    ZombieType.GARGANTUAR: (-20, 0, 80, 150),    # Large hitbox
    ZombieType.GIGA_GARGANTUAR: (-20, 0, 80, 150),
    ZombieType.IMP: (-10, 0, 40, 60),            # Small hitbox
}

# Zombie collision rect (for projectile hits)
ZOMBIE_HURT_RECT = {
    ZombieType.ZOMBIE: (10, 0, 40, 100),
    ZombieType.FLAG: (10, 0, 40, 100),
    ZombieType.CONEHEAD: (10, 0, 40, 110),
    ZombieType.BUCKETHEAD: (10, 0, 40, 110),
    ZombieType.SCREEN_DOOR: (-20, 0, 80, 100),   # Screen door wider
    ZombieType.LADDER: (0, 0, 60, 110),          # Ladder extends
    ZombieType.POLE_VAULTER: (10, 0, 40, 100),
    ZombieType.FOOTBALL: (0, 0, 60, 100),
    ZombieType.ZOMBONI: (-10, 0, 140, 80),       # Vehicle
    ZombieType.CATAPULT: (-10, 0, 140, 80),
    ZombieType.SNORKEL: (10, 0, 40, 100),
    ZombieType.DOLPHIN: (10, 0, 40, 100),
    ZombieType.BALLOON: (10, -20, 40, 80),       # Balloon above
    ZombieType.MINER: (10, 0, 40, 100),
    ZombieType.DANCER: (10, 0, 40, 100),
    ZombieType.BACKUP_DANCER: (10, 0, 40, 100),
    ZombieType.JACK_IN_THE_BOX: (10, 0, 40, 100),
    ZombieType.BUNGEE: (10, 0, 40, 100),
    ZombieType.GARGANTUAR: (0, 0, 100, 150),
    ZombieType.GIGA_GARGANTUAR: (0, 0, 100, 150),
    ZombieType.IMP: (10, 0, 30, 60),
}


# =============================================================================
# PLANT TYPES (The 12 No-Cob Meta Plants + Essentials)
# =============================================================================
class PlantType(IntEnum):
    # Economy
    SUNFLOWER = 0
    TWIN_SUNFLOWER = 1
    # Defense
    PUMPKIN = 2
    GARLIC = 3
    # Support
    LILY_PAD = 4
    COFFEE_BEAN = 5
    IMITATER = 6
    # AOE & Control
    FUME_SHROOM = 7
    GLOOM_SHROOM = 8
    WINTER_MELON = 9
    # Counters / Instants
    SPIKEROCK = 10
    SQUASH = 11
    CHERRY_BOMB = 12
    BLOVER = 13


# =============================================================================
# PLANT STATS
# Format: (hp, cost, recharge_ticks, attack_damage, attack_interval_ticks)
# =============================================================================
PLANT_STATS = {
    # (hp, sun_cost, recharge_ticks, attack_dmg, attack_interval_ticks)
    PlantType.SUNFLOWER: (300, 50, 750, 0, 2400),      # 24s sun production
    PlantType.TWIN_SUNFLOWER: (300, 125, 750, 0, 2400),  # Double sun
    PlantType.PUMPKIN: (4000, 125, 3000, 0, 0),        # Shield layer
    PlantType.GARLIC: (400, 50, 750, 0, 0),            # Row redirect
    PlantType.LILY_PAD: (300, 25, 750, 0, 0),          # Water platform
    PlantType.COFFEE_BEAN: (300, 75, 750, 0, 0),       # Wake mushroom
    PlantType.IMITATER: (300, 0, 750, 0, 0),           # Copy plant
    PlantType.FUME_SHROOM: (300, 75, 750, 20, 149),    # ~1.5s per attack
    PlantType.GLOOM_SHROOM: (300, 150, 750, 20, 190),  # 1.9s per attack, 4x dmg
    PlantType.WINTER_MELON: (300, 500, 750, 80, 300),  # 3s per attack
    PlantType.SPIKEROCK: (600, 125, 750, 20, 10),      # Constant damage
    PlantType.SQUASH: (300, 50, 3000, 1800, 0),        # One-time smash
    PlantType.CHERRY_BOMB: (300, 150, 5000, 1800, 0),  # Instant 3x3
    PlantType.BLOVER: (300, 100, 750, 0, 0),           # Blow away
}

# Plant specific constants
SUNFLOWER_SUN_VALUE = 25
TWIN_SUNFLOWER_SUN_VALUE = 50
GARLIC_REDIRECT_ROWS = 1        # Moves zombie 1 row up or down
FUME_RANGE = 280                # ~3.5 tiles
GLOOM_RANGE = 120               # ~1.5 tiles (3x3 around)
WINTER_MELON_SPLASH_RADIUS = 80  # ~1 tile splash
SPIKEROCK_VEHICLE_HITS = 9       # Can take 9 vehicle hits
SQUASH_JUMP_RANGE = 120          # Detection range
SQUASH_JUMP_DURATION = 30        # Ticks in air
CHERRY_BOMB_RADIUS = 120         # 3x3 area
BLOVER_DURATION = 100            # Effect lasts 1 second


# =============================================================================
# PLANT HITBOXES
# HurtRect: (x_offset, y_offset, width, height) relative to cell position
# =============================================================================
PLANT_HURT_RECT = {
    PlantType.SUNFLOWER: (0, 0, 80, 80),
    PlantType.TWIN_SUNFLOWER: (0, 0, 80, 80),
    PlantType.PUMPKIN: (-10, 0, 100, 90),       # Wider than normal
    PlantType.GARLIC: (0, 0, 60, 70),
    PlantType.LILY_PAD: (0, 0, 80, 20),         # Flat
    PlantType.COFFEE_BEAN: (0, 0, 40, 40),      # Small
    PlantType.IMITATER: (0, 0, 80, 80),
    PlantType.FUME_SHROOM: (0, 0, 80, 80),
    PlantType.GLOOM_SHROOM: (0, 0, 80, 80),
    PlantType.WINTER_MELON: (0, 0, 80, 80),
    PlantType.SPIKEROCK: (0, 0, 80, 40),        # Floor level
    PlantType.SQUASH: (0, 0, 80, 80),
    PlantType.CHERRY_BOMB: (0, 0, 80, 80),
    PlantType.BLOVER: (0, 0, 80, 80),
}


# =============================================================================
# PROJECTILE TYPES
# =============================================================================
class ProjectileType(IntEnum):
    PEA = 0
    SNOW_PEA = 1
    FUME = 2
    SPORE = 3
    WINTERMELON = 4
    BASKETBALL = 5          # Catapult zombie


# =============================================================================
# PROJECTILE STATS
# Format: (damage, speed, is_lobbed, splash_radius, damage_type)
# =============================================================================
PROJECTILE_STATS = {
    # (damage, speed_per_tick, is_lobbed, splash_radius, damage_type)
    ProjectileType.PEA: (20, 3.6, False, 0, DamageType.NORMAL),
    ProjectileType.SNOW_PEA: (20, 3.6, False, 0, DamageType.FREEZE),
    ProjectileType.FUME: (20, 0, False, 0, DamageType.PIERCE),  # Instant hit
    ProjectileType.SPORE: (20, 3.0, False, 0, DamageType.NORMAL),
    ProjectileType.WINTERMELON: (80, 2.5, True, 80, DamageType.FREEZE),
    ProjectileType.BASKETBALL: (75, 3.0, True, 0, DamageType.NORMAL),
}


# =============================================================================
# WAVE & SPAWNING CONSTANTS
# =============================================================================
WAVE_COUNT = 20                     # 20 waves per flag
FLAG_WAVE_INTERVAL = 20             # Flags appear every 20 waves
INITIAL_WAVE_DELAY = 100            # 1 second before first wave
WAVE_SPAWN_DELAY = 2000             # 20 seconds between waves (max)
ZOMBIE_SPAWN_INTERVAL = 50          # 0.5s between zombies in same wave
HUGE_WAVE_ZOMBIE_COUNT = 10         # Extra zombies in huge wave


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def grid_to_pixel(row: int, col: int) -> tuple[float, float]:
    """Convert grid coordinates to pixel coordinates (center of cell)."""
    x = GRID_OFFSET_X + col * CELL_WIDTH + CELL_WIDTH / 2
    y = GRID_OFFSET_Y + row * CELL_HEIGHT + CELL_HEIGHT / 2
    return (x, y)


def pixel_to_grid(x: float, y: float) -> tuple[int, int]:
    """Convert pixel coordinates to grid coordinates."""
    col = int((x - GRID_OFFSET_X) // CELL_WIDTH)
    row = int((y - GRID_OFFSET_Y) // CELL_HEIGHT)
    col = max(0, min(col, NUM_COLS - 1))
    row = max(0, min(row, NUM_ROWS - 1))
    return (row, col)


def is_water_row(row: int) -> bool:
    """Check if a row is a pool (water) row."""
    return row in POOL_ROWS


def get_row_y(row: int) -> float:
    """Get the Y coordinate for the center of a row."""
    return GRID_OFFSET_Y + row * CELL_HEIGHT + CELL_HEIGHT / 2
