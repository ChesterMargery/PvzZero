"""
PvZ Survival Endless Simulator - Main Engine Module
High-fidelity, high-performance headless simulator for Reinforcement Learning.
Target: >10,000 FPS with 100% logic parity.
"""

from typing import Optional
from dataclasses import dataclass, field
import random

from consts import (
    # Types
    ZombieType, ZombieState, PlantType, ProjectileType, DamageType,
    # Zombie constants
    ZOMBIE_STATS, POLE_VAULT_JUMP_DURATION, DOLPHIN_JUMP_DURATION,
    MINER_DIG_TIME, MINER_RISE_STUN, DANCER_SUMMON_CD,
    JACK_EXPLODE_CHANCE, GARG_SMASH_WINDUP, GARG_SMASH_COOLDOWN,
    GARG_THROW_IMP_HP_THRESHOLD, BUNGEE_DESCENT_TIME, BUNGEE_GRAB_TIME,
    BUNGEE_ASCENT_TIME,
    # Plant constants
    PLANT_STATS, SUNFLOWER_SUN_VALUE, TWIN_SUNFLOWER_SUN_VALUE,
    GARLIC_REDIRECT_ROWS, FUME_RANGE, GLOOM_RANGE,
    SPIKEROCK_VEHICLE_HITS, SQUASH_JUMP_DURATION, BLOVER_DURATION,
    # World constants
    TICKS_PER_SECOND, NUM_ROWS, NUM_COLS, ZOMBIE_SPAWN_X,
    CELL_WIDTH, CELL_HEIGHT, GRID_OFFSET_X, GRID_OFFSET_Y,
    POOL_ROWS, CHILL_DURATION,
    # Wave constants
    WAVE_COUNT, INITIAL_WAVE_DELAY, WAVE_SPAWN_DELAY,
    ZOMBIE_SPAWN_INTERVAL, HUGE_WAVE_ZOMBIE_COUNT,
    # Helpers
    grid_to_pixel, pixel_to_grid, is_water_row, get_row_y,
)
from state import Zombie, Plant, Projectile, Sun, LawnMower, Rect
from physics import (
    ProjectileManager, CollisionSystem, FumeAttack, GloomAttack,
    CherryBombAttack, SquashAttack, SpikerockCollision, GarlicCollision,
    LadderPlacement, check_collision, check_radius_collision, distance,
)


@dataclass
class GameState:
    """Complete game state snapshot for RL observations."""
    tick: int
    sun: int
    wave: int
    zombies_remaining: int
    plants_alive: int
    lawn_mowers_left: int
    game_over: bool
    victory: bool


class PvZSim:
    """
    High-fidelity Plants vs. Zombies Survival Endless simulator.
    
    Implements 1:1 physics replica of the original game.
    Optimized for >10,000 FPS headless simulation.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the simulator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        
        # Game clock
        self.tick: int = 0
        
        # Economy
        self.sun: int = 50  # Starting sun
        
        # Entities
        self.zombies: list[Zombie] = []
        self.plants: list[Plant] = []
        self.lawn_mowers: list[LawnMower] = []
        self.suns: list[Sun] = []
        
        # ID counters
        self._next_zombie_id: int = 0
        self._next_plant_id: int = 0
        self._next_sun_id: int = 0
        
        # Projectile system
        self.projectile_manager = ProjectileManager()
        
        # Wave system
        self.wave: int = 0
        self.wave_timer: int = INITIAL_WAVE_DELAY
        self.zombies_to_spawn: list[tuple[ZombieType, int]] = []  # (type, row)
        self.spawn_timer: int = 0
        
        # Game state
        self.game_over: bool = False
        self.victory: bool = False
        
        # Plant grid cache for O(1) lookup
        self._plant_grid: list[list[Optional[Plant]]] = [
            [None for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)
        ]
        self._pumpkin_grid: list[list[Optional[Plant]]] = [
            [None for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)
        ]
        
        # Blover active timer
        self._blover_timer: int = 0
        
        # Initialize lawn mowers
        self._init_lawn_mowers()
    
    def _init_lawn_mowers(self) -> None:
        """Initialize lawn mowers for each row."""
        for row in range(NUM_ROWS):
            x = GRID_OFFSET_X - 40  # Left of first column
            self.lawn_mowers.append(LawnMower(mRow=row, mX=x))
    
    def reset(self, seed: Optional[int] = None) -> GameState:
        """Reset the simulation to initial state."""
        if seed is not None:
            random.seed(seed)
        
        self.tick = 0
        self.sun = 50
        
        self.zombies.clear()
        self.plants.clear()
        self.lawn_mowers.clear()
        self.suns.clear()
        
        self._next_zombie_id = 0
        self._next_plant_id = 0
        self._next_sun_id = 0
        
        self.projectile_manager.clear()
        
        self.wave = 0
        self.wave_timer = INITIAL_WAVE_DELAY
        self.zombies_to_spawn.clear()
        self.spawn_timer = 0
        
        self.game_over = False
        self.victory = False
        
        self._plant_grid = [
            [None for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)
        ]
        self._pumpkin_grid = [
            [None for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)
        ]
        
        self._blover_timer = 0
        
        self._init_lawn_mowers()
        
        return self.get_state()
    
    def get_state(self) -> GameState:
        """Get current game state snapshot."""
        return GameState(
            tick=self.tick,
            sun=self.sun,
            wave=self.wave,
            zombies_remaining=len([z for z in self.zombies if z.is_alive()]),
            plants_alive=len([p for p in self.plants if p.is_alive()]),
            lawn_mowers_left=len([m for m in self.lawn_mowers if m.mIsActive]),
            game_over=self.game_over,
            victory=self.victory,
        )
    
    # =========================================================================
    # ENTITY SPAWNING
    # =========================================================================
    
    def spawn_zombie(self, zombie_type: ZombieType, row: int, 
                     x: Optional[float] = None) -> Zombie:
        """
        Spawn a zombie at the specified row.
        
        Args:
            zombie_type: Type of zombie to spawn
            row: Row (0-5) to spawn in
            x: Optional X position (default: right edge)
            
        Returns:
            The spawned Zombie entity
        """
        if x is None:
            x = ZOMBIE_SPAWN_X
        
        y = get_row_y(row)
        
        zombie = Zombie.create(
            self._next_zombie_id,
            zombie_type,
            row,
            x,
            y
        )
        self._next_zombie_id += 1
        self.zombies.append(zombie)
        
        return zombie
    
    def plant_at(self, plant_type: PlantType, row: int, col: int,
                 free: bool = False) -> Optional[Plant]:
        """
        Plant a plant at the specified grid position.
        
        Args:
            plant_type: Type of plant to place
            row: Row (0-5)
            col: Column (0-8)
            free: If True, doesn't cost sun
            
        Returns:
            The planted Plant entity, or None if invalid placement
        """
        # Validate position
        if not (0 <= row < NUM_ROWS and 0 <= col < NUM_COLS):
            return None
        
        # Check water requirement
        is_water = is_water_row(row)
        
        # Lily Pad required for water (except Lily Pad itself)
        if is_water and plant_type != PlantType.LILY_PAD:
            existing = self._plant_grid[row][col]
            if existing is None or existing.mPlantType != PlantType.LILY_PAD:
                return None
        
        # Check if slot is occupied
        if plant_type == PlantType.PUMPKIN:
            # Pumpkin can stack on existing plants
            if self._pumpkin_grid[row][col] is not None:
                return None
            if self._plant_grid[row][col] is None:
                return None  # Need a plant to protect
        elif plant_type == PlantType.LILY_PAD:
            if self._plant_grid[row][col] is not None:
                return None
        else:
            # Check if there's already a non-lily-pad plant
            existing = self._plant_grid[row][col]
            if existing is not None and existing.mPlantType != PlantType.LILY_PAD:
                return None
        
        # Check sun cost
        stats = PLANT_STATS[plant_type]
        cost = stats[1]
        
        if not free and self.sun < cost:
            return None
        
        if not free:
            self.sun -= cost
        
        # Create plant
        plant = Plant.create(self._next_plant_id, plant_type, row, col)
        self._next_plant_id += 1
        self.plants.append(plant)
        
        # Update grid cache
        if plant_type == PlantType.PUMPKIN:
            self._pumpkin_grid[row][col] = plant
            # Add pumpkin HP to existing plant
            existing = self._plant_grid[row][col]
            if existing:
                existing.add_pumpkin()
        else:
            if self._plant_grid[row][col] is None:
                self._plant_grid[row][col] = plant
            elif self._plant_grid[row][col].mPlantType == PlantType.LILY_PAD:
                # Plant on top of lily pad
                self._plant_grid[row][col] = plant
        
        return plant
    
    def remove_plant(self, plant: Plant) -> None:
        """Remove a plant from the game."""
        plant.mIsActive = False
        plant.mHP = 0
        
        row, col = plant.mRow, plant.mCol
        
        if plant.mPlantType == PlantType.PUMPKIN:
            self._pumpkin_grid[row][col] = None
        else:
            if self._plant_grid[row][col] == plant:
                self._plant_grid[row][col] = None
    
    def add_sun(self, amount: int) -> None:
        """Add sun to the player's economy."""
        self.sun += amount
    
    # =========================================================================
    # MAIN SIMULATION LOOP
    # =========================================================================
    
    def step(self) -> GameState:
        """
        Advance the simulation by one tick (1/100th of a second).
        
        This is the main game loop that updates all game systems:
        1. Update timers and spawn waves
        2. Update zombie state machines
        3. Update plant attacks
        4. Update projectiles and collisions
        5. Process damage and deaths
        6. Check win/lose conditions
        
        Returns:
            Current game state after this tick
        """
        if self.game_over:
            return self.get_state()
        
        self.tick += 1
        
        # Update blover timer
        if self._blover_timer > 0:
            self._blover_timer -= 1
        
        # 1. Update wave spawning
        self._update_waves()
        
        # 2. Update zombies
        self._update_zombies()
        
        # 3. Update plants
        self._update_plants()
        
        # 4. Update projectiles
        hits = self.projectile_manager.update(self.zombies)
        for zombie, damage, dmg_type in hits:
            zombie.take_damage(damage, dmg_type)
        
        # 5. Process collisions and damage
        self._process_collisions()
        
        # 6. Update lawn mowers
        self._update_lawn_mowers()
        
        # 7. Clean up dead entities
        self._cleanup()
        
        # 8. Check win/lose conditions
        self._check_game_end()
        
        return self.get_state()
    
    def _update_waves(self) -> None:
        """Update wave timer and spawn zombies."""
        # Spawn queued zombies
        if self.zombies_to_spawn and self.spawn_timer <= 0:
            zombie_type, row = self.zombies_to_spawn.pop(0)
            self.spawn_zombie(zombie_type, row)
            self.spawn_timer = ZOMBIE_SPAWN_INTERVAL
        
        if self.spawn_timer > 0:
            self.spawn_timer -= 1
        
        # Wave timer countdown
        if self.wave_timer > 0:
            self.wave_timer -= 1
        elif not self.zombies_to_spawn:
            # Start new wave when timer expires and no pending spawns
            self._start_new_wave()
    
    def _start_new_wave(self) -> None:
        """Start a new wave of zombies."""
        self.wave += 1
        
        # Reset wave timer (varies based on remaining zombies)
        alive_zombies = len([z for z in self.zombies if z.is_alive()])
        if alive_zombies > 10:
            self.wave_timer = WAVE_SPAWN_DELAY
        else:
            self.wave_timer = WAVE_SPAWN_DELAY // 2
        
        # Generate wave composition
        self._generate_wave()
    
    def _generate_wave(self) -> None:
        """Generate zombie composition for current wave."""
        # Base zombie count scales with wave
        base_count = min(3 + self.wave // 2, 15)
        
        # Determine zombie types available based on wave
        available_types = self._get_available_zombie_types()
        
        # Generate spawn list
        for _ in range(base_count):
            zombie_type = random.choice(available_types)
            row = random.randint(0, NUM_ROWS - 1)
            
            # Water zombies only in pool rows
            if zombie_type in (ZombieType.SNORKEL, ZombieType.DOLPHIN):
                row = random.choice(POOL_ROWS)
            
            self.zombies_to_spawn.append((zombie_type, row))
        
        # Special wave handling (huge waves, flag waves)
        if self.wave % WAVE_COUNT == 0:  # Flag wave
            # Add more zombies
            for _ in range(HUGE_WAVE_ZOMBIE_COUNT):
                zombie_type = random.choice(available_types)
                row = random.randint(0, NUM_ROWS - 1)
                self.zombies_to_spawn.append((zombie_type, row))
            
            # Guarantee a Gargantuar on later waves
            if self.wave >= 20:
                garg_type = (ZombieType.GIGA_GARGANTUAR 
                            if self.wave >= 40 
                            else ZombieType.GARGANTUAR)
                row = random.randint(0, NUM_ROWS - 1)
                self.zombies_to_spawn.append((garg_type, row))
    
    def _get_available_zombie_types(self) -> list[ZombieType]:
        """Get zombie types available for current wave."""
        types = [ZombieType.ZOMBIE, ZombieType.CONEHEAD, ZombieType.BUCKETHEAD]
        
        if self.wave >= 2:
            types.append(ZombieType.FLAG)
        if self.wave >= 3:
            types.extend([ZombieType.POLE_VAULTER, ZombieType.SNORKEL])
        if self.wave >= 5:
            types.extend([ZombieType.SCREEN_DOOR, ZombieType.DOLPHIN])
        if self.wave >= 7:
            types.extend([ZombieType.FOOTBALL, ZombieType.BALLOON])
        if self.wave >= 10:
            types.extend([ZombieType.LADDER, ZombieType.JACK_IN_THE_BOX])
        if self.wave >= 12:
            types.extend([ZombieType.MINER, ZombieType.ZOMBONI])
        if self.wave >= 15:
            types.extend([ZombieType.CATAPULT, ZombieType.DANCER])
        if self.wave >= 20:
            types.append(ZombieType.GARGANTUAR)
        if self.wave >= 40:
            types.append(ZombieType.GIGA_GARGANTUAR)
        
        return types
    
    # =========================================================================
    # ZOMBIE STATE MACHINE
    # =========================================================================
    
    def _update_zombies(self) -> None:
        """Update all zombie state machines."""
        for zombie in self.zombies:
            if not zombie.is_alive():
                continue
            
            # Update status effects
            zombie.mStatus.tick()
            
            # Update state timer
            zombie.mStateTimer += 1
            
            # Handle specific zombie types
            handler = self._zombie_handlers.get(zombie.mZombieType, 
                                                 self._update_basic_zombie)
            handler(self, zombie)
    
    def _update_basic_zombie(self, zombie: Zombie) -> None:
        """Update basic zombie (and similar types) state machine."""
        if zombie.mStatus.is_immobilized:
            return
        
        if zombie.mState == ZombieState.WALK:
            # Move towards house
            speed = zombie.get_effective_speed()
            zombie.move(-speed)
            
        elif zombie.mState == ZombieState.EAT:
            # Attack timer
            zombie.mAttackTimer -= 1
            if zombie.mAttackTimer <= 0:
                # Reset attack timer
                zombie.mAttackTimer = zombie.get_effective_attack_interval()
                # Damage will be applied in collision phase
    
    def _update_pole_vaulter(self, zombie: Zombie) -> None:
        """Update Pole Vaulter zombie state machine."""
        if zombie.mStatus.is_immobilized:
            return
        
        if zombie.mState == ZombieState.WALK:
            speed = zombie.get_effective_speed()
            
            if not zombie.mHasJumped:
                # Running fast before jump
                zombie.move(-speed)
            else:
                # Slow walk after jump
                zombie.move(-speed * 0.5)
                
        elif zombie.mState == ZombieState.JUMPING:
            # During jump, move forward faster
            zombie.move(-zombie.mSpeed * 1.5)
            
            if zombie.mStateTimer >= POLE_VAULT_JUMP_DURATION:
                zombie.mState = ZombieState.WALK
                zombie.mHasJumped = True
                zombie.mStateTimer = 0
                
        elif zombie.mState == ZombieState.EAT:
            zombie.mAttackTimer -= 1
            if zombie.mAttackTimer <= 0:
                zombie.mAttackTimer = zombie.get_effective_attack_interval()
    
    def _update_football_zombie(self, zombie: Zombie) -> None:
        """Update Football zombie - same as basic but faster."""
        self._update_basic_zombie(zombie)
    
    def _update_zomboni(self, zombie: Zombie) -> None:
        """Update Zomboni (Ice Resurfacing Car) state machine."""
        if zombie.mState == ZombieState.DRIVING:
            # Zombonis don't get slowed by chill/freeze as much
            speed = zombie.mSpeed * (1.0 if zombie.mStatus.is_frozen else 1.0)
            zombie.move(-speed)
            
            # Zomboni crushes plants (handled in collision)
    
    def _update_catapult_zombie(self, zombie: Zombie) -> None:
        """Update Catapult (Basketball) zombie state machine."""
        if zombie.mState == ZombieState.DRIVING:
            # Drive until reaching target column
            target_x = GRID_OFFSET_X + 7 * CELL_WIDTH  # Stop at column 7
            
            if zombie.mX > target_x:
                zombie.move(-zombie.mSpeed)
            else:
                zombie.mState = ZombieState.EAT  # Start throwing
                zombie.mAttackTimer = zombie.mAttackInterval
                
        elif zombie.mState == ZombieState.EAT:
            zombie.mAttackTimer -= 1
            
            if zombie.mAttackTimer <= 0:
                # Throw basketball at rear plant
                self._catapult_throw(zombie)
                zombie.mAttackTimer = zombie.mAttackInterval
    
    def _catapult_throw(self, zombie: Zombie) -> None:
        """Catapult zombie throws a basketball."""
        # Find rearmost plant in row
        target_plant: Optional[Plant] = None
        min_col = -1
        
        for plant in self.plants:
            if plant.is_alive() and plant.mRow == zombie.mRow:
                if plant.mCol > min_col:
                    min_col = plant.mCol
                    target_plant = plant
        
        if target_plant:
            # Create basketball projectile
            self.projectile_manager.create_projectile(
                ProjectileType.BASKETBALL,
                zombie.mX, zombie.mY, zombie.mRow,
                target_plant.mX, target_plant.mY
            )
    
    def _update_snorkel_zombie(self, zombie: Zombie) -> None:
        """Update Snorkel zombie state machine."""
        if zombie.mStatus.is_immobilized:
            return
        
        if zombie.mState == ZombieState.SUBMERGED:
            # Move underwater (immune to most attacks)
            zombie.move(-zombie.get_effective_speed())
            zombie.mIsSubmerged = True
            
        elif zombie.mState == ZombieState.WALK:
            zombie.mIsSubmerged = False
            zombie.move(-zombie.get_effective_speed())
            
        elif zombie.mState == ZombieState.EAT:
            zombie.mIsSubmerged = False
            zombie.mAttackTimer -= 1
            if zombie.mAttackTimer <= 0:
                zombie.mAttackTimer = zombie.get_effective_attack_interval()
    
    def _update_dolphin_zombie(self, zombie: Zombie) -> None:
        """Update Dolphin Rider zombie state machine."""
        if zombie.mStatus.is_immobilized:
            return
        
        if zombie.mState == ZombieState.WALK:
            speed = zombie.get_effective_speed()
            
            if not zombie.mHasJumped:
                # Fast with dolphin
                zombie.move(-speed)
            else:
                # Slow walk after losing dolphin
                zombie.move(-speed * 0.3)
                
        elif zombie.mState == ZombieState.JUMPING:
            zombie.move(-zombie.mSpeed * 2.0)
            
            if zombie.mStateTimer >= DOLPHIN_JUMP_DURATION:
                zombie.mState = ZombieState.WALK
                zombie.mHasJumped = True
                zombie.mStateTimer = 0
                
        elif zombie.mState == ZombieState.EAT:
            zombie.mAttackTimer -= 1
            if zombie.mAttackTimer <= 0:
                zombie.mAttackTimer = zombie.get_effective_attack_interval()
    
    def _update_balloon_zombie(self, zombie: Zombie) -> None:
        """Update Balloon zombie state machine."""
        if zombie.mState == ZombieState.FLYING:
            zombie.mIsFlying = True
            zombie.move(-zombie.get_effective_speed())
            
            # Check if blown away by Blover
            if self._blover_timer > 0:
                zombie.mState = ZombieState.DYING
                zombie.mHP = 0
                
        elif zombie.mState == ZombieState.WALK:
            # Balloon popped, walking
            zombie.mIsFlying = False
            zombie.move(-zombie.get_effective_speed())
            
        elif zombie.mState == ZombieState.EAT:
            zombie.mIsFlying = False
            zombie.mAttackTimer -= 1
            if zombie.mAttackTimer <= 0:
                zombie.mAttackTimer = zombie.get_effective_attack_interval()
    
    def _update_miner_zombie(self, zombie: Zombie) -> None:
        """Update Digger/Miner zombie state machine."""
        if zombie.mState == ZombieState.SPECIAL:
            # Digging underground
            zombie.mIsDigging = True
            zombie.mStateTimer += 1
            
            if zombie.mStateTimer >= MINER_DIG_TIME:
                # Surface at left side
                zombie.mState = ZombieState.RISING
                zombie.mStateTimer = 0
                zombie.mX = GRID_OFFSET_X - 20  # Left of grid
                zombie.invalidate_rects()
                
        elif zombie.mState == ZombieState.RISING:
            zombie.mIsDigging = False
            zombie.mStateTimer += 1
            
            if zombie.mStateTimer >= MINER_RISE_STUN:
                zombie.mState = ZombieState.WALK
                zombie.mStateTimer = 0
                
        elif zombie.mState == ZombieState.WALK:
            # Miner walks RIGHT after surfacing (eats from behind)
            zombie.move(zombie.get_effective_speed())
            
        elif zombie.mState == ZombieState.EAT:
            zombie.mAttackTimer -= 1
            if zombie.mAttackTimer <= 0:
                zombie.mAttackTimer = zombie.get_effective_attack_interval()
    
    def _update_dancer_zombie(self, zombie: Zombie) -> None:
        """Update Dancing zombie state machine."""
        if zombie.mStatus.is_immobilized:
            return
        
        if zombie.mState == ZombieState.WALK:
            zombie.move(-zombie.get_effective_speed())
            
            zombie.mSummonTimer += 1
            if zombie.mSummonTimer >= DANCER_SUMMON_CD:
                zombie.mState = ZombieState.DANCING
                zombie.mStateTimer = 0
                zombie.mSummonTimer = 0
                
        elif zombie.mState == ZombieState.DANCING:
            if zombie.mStateTimer == 30:  # Summon at animation peak
                self._summon_backup_dancers(zombie)
            
            if zombie.mStateTimer >= 60:
                zombie.mState = ZombieState.WALK
                zombie.mStateTimer = 0
                
        elif zombie.mState == ZombieState.EAT:
            zombie.mAttackTimer -= 1
            if zombie.mAttackTimer <= 0:
                zombie.mAttackTimer = zombie.get_effective_attack_interval()
    
    def _summon_backup_dancers(self, zombie: Zombie) -> None:
        """Summon 4 backup dancers in cross formation around dancer."""
        offsets = [
            (0, -CELL_HEIGHT),   # Up
            (0, CELL_HEIGHT),    # Down
            (-CELL_WIDTH, 0),    # Left
            (CELL_WIDTH, 0),     # Right
        ]
        
        for dx, dy in offsets:
            new_row = zombie.mRow
            if dy < 0 and zombie.mRow > 0:
                new_row = zombie.mRow - 1
            elif dy > 0 and zombie.mRow < NUM_ROWS - 1:
                new_row = zombie.mRow + 1
            else:
                continue
            
            backup = Zombie.create(
                self._next_zombie_id,
                ZombieType.BACKUP_DANCER,
                new_row,
                zombie.mX + dx,
                get_row_y(new_row)
            )
            self._next_zombie_id += 1
            self.zombies.append(backup)
    
    def _update_jack_in_the_box(self, zombie: Zombie) -> None:
        """Update Jack-in-the-box zombie state machine."""
        if zombie.mStatus.is_immobilized:
            return
        
        if zombie.mState == ZombieState.WALK:
            zombie.move(-zombie.get_effective_speed())
            
            # Random chance to explode
            if random.random() < JACK_EXPLODE_CHANCE:
                zombie.mState = ZombieState.EXPLODING
                zombie.mStateTimer = 0
                
        elif zombie.mState == ZombieState.EXPLODING:
            if zombie.mStateTimer >= 20:  # Short delay
                # Explode! Kill all plants in 3x3 area
                self._jack_explode(zombie)
                zombie.mState = ZombieState.DEAD
                zombie.mHP = 0
                
        elif zombie.mState == ZombieState.EAT:
            zombie.mAttackTimer -= 1
            if zombie.mAttackTimer <= 0:
                zombie.mAttackTimer = zombie.get_effective_attack_interval()
    
    def _jack_explode(self, zombie: Zombie) -> None:
        """Jack-in-the-box explosion - kills plants in 3x3."""
        for plant in self.plants:
            if not plant.is_alive():
                continue
            
            dist = distance(zombie.mX, zombie.mY, plant.mX, plant.mY)
            if dist <= 120:  # 3x3 radius
                plant.mHP = 0
                plant.mIsActive = False
                self.remove_plant(plant)
    
    def _update_bungee_zombie(self, zombie: Zombie) -> None:
        """Update Bungee zombie state machine."""
        if zombie.mState == ZombieState.DESCENDING:
            zombie.mStateTimer += 1
            
            if zombie.mStateTimer >= BUNGEE_DESCENT_TIME:
                zombie.mState = ZombieState.GRABBING
                zombie.mStateTimer = 0
                
        elif zombie.mState == ZombieState.GRABBING:
            zombie.mStateTimer += 1
            
            if zombie.mStateTimer == 1:
                # Grab plant at current position
                row, col = pixel_to_grid(zombie.mX, zombie.mY)
                plant = self._plant_grid[row][col]
                if plant and plant.is_alive():
                    zombie.mTargetPlantId = plant.mPlantId
            
            if zombie.mStateTimer >= BUNGEE_GRAB_TIME:
                zombie.mState = ZombieState.ASCENDING
                zombie.mStateTimer = 0
                
                # Kill grabbed plant
                if zombie.mTargetPlantId >= 0:
                    for plant in self.plants:
                        if plant.mPlantId == zombie.mTargetPlantId:
                            self.remove_plant(plant)
                            break
                
        elif zombie.mState == ZombieState.ASCENDING:
            zombie.mStateTimer += 1
            
            if zombie.mStateTimer >= BUNGEE_ASCENT_TIME:
                zombie.mState = ZombieState.DEAD
    
    def _update_gargantuar(self, zombie: Zombie) -> None:
        """Update Gargantuar/Giga-Gargantuar state machine."""
        if zombie.mStatus.is_immobilized:
            return
        
        if zombie.mState == ZombieState.WALK:
            zombie.move(-zombie.get_effective_speed())
            
            # Check if should throw imp
            if not zombie.mHasThrownImp:
                hp_ratio = zombie.mHP / zombie.mMaxHP
                if hp_ratio <= GARG_THROW_IMP_HP_THRESHOLD:
                    self._garg_throw_imp(zombie)
                    zombie.mHasThrownImp = True
                    
        elif zombie.mState == ZombieState.SMASHING:
            zombie.mStateTimer += 1
            
            if zombie.mStateTimer >= GARG_SMASH_WINDUP:
                # SMASH! Instant kill target plant
                if zombie.mTargetPlantId >= 0:
                    for plant in self.plants:
                        if plant.mPlantId == zombie.mTargetPlantId:
                            self.remove_plant(plant)
                            break
                
                zombie.mState = ZombieState.WALK
                zombie.mStateTimer = 0
                zombie.mAttackTimer = GARG_SMASH_COOLDOWN
                
        elif zombie.mState == ZombieState.EAT:
            # Gargantuars don't eat, they smash
            zombie.mState = ZombieState.WALK
    
    def _garg_throw_imp(self, zombie: Zombie) -> None:
        """Gargantuar throws Imp ahead."""
        # Throw imp 3-5 columns ahead
        throw_distance = random.randint(3, 5) * CELL_WIDTH
        
        imp = Zombie.create(
            self._next_zombie_id,
            ZombieType.IMP,
            zombie.mRow,
            zombie.mX - throw_distance,
            zombie.mY
        )
        self._next_zombie_id += 1
        self.zombies.append(imp)
    
    # Zombie handler dispatch table
    _zombie_handlers = {
        ZombieType.ZOMBIE: _update_basic_zombie,
        ZombieType.FLAG: _update_basic_zombie,
        ZombieType.CONEHEAD: _update_basic_zombie,
        ZombieType.BUCKETHEAD: _update_basic_zombie,
        ZombieType.SCREEN_DOOR: _update_basic_zombie,
        ZombieType.LADDER: _update_basic_zombie,
        ZombieType.POLE_VAULTER: _update_pole_vaulter,
        ZombieType.FOOTBALL: _update_football_zombie,
        ZombieType.ZOMBONI: _update_zomboni,
        ZombieType.CATAPULT: _update_catapult_zombie,
        ZombieType.SNORKEL: _update_snorkel_zombie,
        ZombieType.DOLPHIN: _update_dolphin_zombie,
        ZombieType.BALLOON: _update_balloon_zombie,
        ZombieType.MINER: _update_miner_zombie,
        ZombieType.DANCER: _update_dancer_zombie,
        ZombieType.BACKUP_DANCER: _update_basic_zombie,
        ZombieType.JACK_IN_THE_BOX: _update_jack_in_the_box,
        ZombieType.BUNGEE: _update_bungee_zombie,
        ZombieType.GARGANTUAR: _update_gargantuar,
        ZombieType.GIGA_GARGANTUAR: _update_gargantuar,
        ZombieType.IMP: _update_basic_zombie,
    }
    
    # =========================================================================
    # PLANT UPDATES
    # =========================================================================
    
    def _update_plants(self) -> None:
        """Update all plant behaviors."""
        for plant in self.plants:
            if not plant.is_alive():
                continue
            
            # Update state timer
            plant.mStateTimer += 1
            
            # Handle specific plant types
            handler = self._plant_handlers.get(plant.mPlantType,
                                                self._update_basic_plant)
            handler(self, plant)
    
    def _update_basic_plant(self, plant: Plant) -> None:
        """Default plant update (no special behavior)."""
        pass
    
    def _update_sunflower(self, plant: Plant) -> None:
        """Update Sunflower sun production."""
        plant.mSunTimer -= 1
        
        if plant.mSunTimer <= 0:
            self.sun += SUNFLOWER_SUN_VALUE
            plant.mSunTimer = plant.mAttackInterval
    
    def _update_twin_sunflower(self, plant: Plant) -> None:
        """Update Twin Sunflower sun production."""
        plant.mSunTimer -= 1
        
        if plant.mSunTimer <= 0:
            self.sun += TWIN_SUNFLOWER_SUN_VALUE
            plant.mSunTimer = plant.mAttackInterval
    
    def _update_fume_shroom(self, plant: Plant) -> None:
        """Update Fume-shroom attack."""
        if not plant.mIsAwake:
            return
        
        plant.mAttackTimer -= 1
        
        if plant.mAttackTimer <= 0:
            # Check for zombies in range
            targets = FumeAttack.get_targets(plant, self.zombies)
            
            if targets:
                # Apply damage to all targets (pierce shield)
                for zombie in targets:
                    zombie.take_damage(plant.mAttackDamage, DamageType.PIERCE)
                
                plant.mAttackTimer = plant.mAttackInterval
    
    def _update_gloom_shroom(self, plant: Plant) -> None:
        """Update Gloom-shroom 3x3 AOE attack."""
        if not plant.mIsAwake:
            return
        
        plant.mAttackTimer -= 1
        
        if plant.mAttackTimer <= 0:
            # Check for zombies in 3x3 range
            targets = GloomAttack.get_targets(plant, self.zombies)
            
            if targets:
                # Apply damage to all targets (4x damage, pierce shield)
                for zombie in targets:
                    zombie.take_damage(plant.mAttackDamage * 4, DamageType.PIERCE)
                
                plant.mAttackTimer = plant.mAttackInterval
    
    def _update_winter_melon(self, plant: Plant) -> None:
        """Update Winter Melon lobbed attack."""
        plant.mAttackTimer -= 1
        
        if plant.mAttackTimer <= 0:
            # Find target zombie in lane
            target = CollisionSystem.get_frontmost_zombie_in_lane(
                plant.mRow, self.zombies
            )
            
            if target:
                # Create winter melon projectile
                self.projectile_manager.create_projectile(
                    ProjectileType.WINTERMELON,
                    plant.mX, plant.mY, plant.mRow,
                    target.mX, target.mY,
                    plant.mPlantId
                )
                plant.mAttackTimer = plant.mAttackInterval
    
    def _update_spikerock(self, plant: Plant) -> None:
        """Update Spikerock constant damage."""
        # Damage is applied in collision phase
        pass
    
    def _update_squash(self, plant: Plant) -> None:
        """Update Squash detection and jump."""
        if plant.mSquashJumping:
            plant.mStateTimer += 1
            
            if plant.mStateTimer >= SQUASH_JUMP_DURATION:
                # Land and smash
                target_zombie: Optional[Zombie] = None
                for zombie in self.zombies:
                    if zombie.mZombieId == plant.mSquashTargetId:
                        target_zombie = zombie
                        break
                
                if target_zombie and target_zombie.is_alive():
                    target_zombie.take_damage(plant.mAttackDamage, DamageType.EXPLOSION)
                
                # Squash dies after smash
                self.remove_plant(plant)
        else:
            # Look for targets
            target = SquashAttack.find_target(plant, self.zombies)
            
            if target:
                plant.mSquashJumping = True
                plant.mSquashTargetId = target.mZombieId
                plant.mStateTimer = 0
    
    def _update_cherry_bomb(self, plant: Plant) -> None:
        """Update Cherry Bomb instant explosion."""
        if plant.mStateTimer >= 1:  # Explode after 1 tick
            targets = CherryBombAttack.get_targets(plant, self.zombies)
            
            for zombie in targets:
                zombie.take_damage(plant.mAttackDamage, DamageType.EXPLOSION)
            
            self.remove_plant(plant)
    
    def _update_blover(self, plant: Plant) -> None:
        """Update Blover wind effect."""
        if plant.mStateTimer >= 1:
            self._blover_timer = BLOVER_DURATION
            self.remove_plant(plant)
    
    def _update_garlic(self, plant: Plant) -> None:
        """Garlic redirect is handled in collision phase."""
        pass
    
    def _update_coffee_bean(self, plant: Plant) -> None:
        """Update Coffee Bean mushroom wake effect."""
        if plant.mStateTimer >= 1:
            # Wake up mushroom at same position
            target_plant = self._plant_grid[plant.mRow][plant.mCol]
            if target_plant and target_plant != plant:
                target_plant.wake_up()
            
            self.remove_plant(plant)
    
    # Plant handler dispatch table
    _plant_handlers = {
        PlantType.SUNFLOWER: _update_sunflower,
        PlantType.TWIN_SUNFLOWER: _update_twin_sunflower,
        PlantType.PUMPKIN: _update_basic_plant,
        PlantType.GARLIC: _update_garlic,
        PlantType.LILY_PAD: _update_basic_plant,
        PlantType.COFFEE_BEAN: _update_coffee_bean,
        PlantType.IMITATER: _update_basic_plant,
        PlantType.FUME_SHROOM: _update_fume_shroom,
        PlantType.GLOOM_SHROOM: _update_gloom_shroom,
        PlantType.WINTER_MELON: _update_winter_melon,
        PlantType.SPIKEROCK: _update_spikerock,
        PlantType.SQUASH: _update_squash,
        PlantType.CHERRY_BOMB: _update_cherry_bomb,
        PlantType.BLOVER: _update_blover,
    }
    
    # =========================================================================
    # COLLISION PROCESSING
    # =========================================================================
    
    def _process_collisions(self) -> None:
        """Process all zombie-plant collisions."""
        # Get all collision pairs
        collisions = CollisionSystem.get_zombie_plant_collisions(
            self.zombies, self.plants
        )
        
        # Process each collision
        for zombie, plant in collisions:
            self._handle_zombie_plant_collision(zombie, plant)
        
        # Process spikerock damage
        self._process_spikerock_damage()
    
    def _handle_zombie_plant_collision(self, zombie: Zombie, plant: Plant) -> None:
        """Handle a single zombie-plant collision."""
        # Special cases first
        
        # Pole Vaulter jump
        if (zombie.mZombieType == ZombieType.POLE_VAULTER and 
            not zombie.mHasJumped and 
            zombie.mState == ZombieState.WALK):
            zombie.mState = ZombieState.JUMPING
            zombie.mStateTimer = 0
            return
        
        # Dolphin jump
        if (zombie.mZombieType == ZombieType.DOLPHIN and 
            not zombie.mHasJumped and 
            zombie.mState == ZombieState.WALK):
            zombie.mState = ZombieState.JUMPING
            zombie.mStateTimer = 0
            return
        
        # Ladder placement
        if LadderPlacement.can_place_ladder(zombie, plant):
            zombie.mState = ZombieState.PLACING_LADDER
            zombie.mHasLadder = False
            zombie.mStateTimer = 0
            # Ladder zombie becomes faster after placing
            zombie.mSpeed *= 1.5
            return
        
        # Gargantuar smash
        if zombie.mZombieType in (ZombieType.GARGANTUAR, ZombieType.GIGA_GARGANTUAR):
            if zombie.mState == ZombieState.WALK and zombie.mAttackTimer <= 0:
                zombie.mState = ZombieState.SMASHING
                zombie.mTargetPlantId = plant.mPlantId
                zombie.mStateTimer = 0
            return
        
        # Zomboni crush
        if zombie.mZombieType == ZombieType.ZOMBONI:
            if plant.mPlantType == PlantType.SPIKEROCK:
                # Spikerock damages Zomboni
                plant.mVehicleHits += 1
                if plant.mVehicleHits >= SPIKEROCK_VEHICLE_HITS:
                    self.remove_plant(plant)
                zombie.take_damage(plant.mAttackDamage * 3)
            else:
                # Zomboni crushes plant
                self.remove_plant(plant)
            return
        
        # Garlic redirect
        if plant.mPlantType == PlantType.GARLIC:
            if zombie.mState == ZombieState.WALK:
                zombie.mState = ZombieState.EAT
                zombie.mAttackTimer = zombie.mAttackInterval
            
            zombie.mAttackTimer -= 1
            if zombie.mAttackTimer <= 0:
                # Apply damage to garlic
                plant.take_damage(zombie.mAttackDamage)
                zombie.mAttackTimer = zombie.mAttackInterval
                
                # Redirect zombie
                new_row = GarlicCollision.redirect_zombie(zombie, zombie.mRow)
                if new_row != zombie.mRow:
                    zombie.mRow = new_row
                    zombie.mY = get_row_y(new_row)
                    zombie.invalidate_rects()
                    zombie.mState = ZombieState.WALK
            return
        
        # Snorkel surfaces to eat
        if zombie.mZombieType == ZombieType.SNORKEL and zombie.mIsSubmerged:
            zombie.mState = ZombieState.EAT
            zombie.mIsSubmerged = False
        
        # Regular eating
        if zombie.mState == ZombieState.WALK:
            zombie.mState = ZombieState.EAT
            zombie.mAttackTimer = 0
        
        if zombie.mState == ZombieState.EAT and zombie.mAttackTimer <= 0:
            # Apply damage to plant
            plant.take_damage(zombie.mAttackDamage)
            zombie.mAttackTimer = zombie.get_effective_attack_interval()
            
            if not plant.is_alive():
                zombie.mState = ZombieState.WALK
    
    def _process_spikerock_damage(self) -> None:
        """Apply spikerock floor damage to zombies."""
        for plant in self.plants:
            if not plant.is_alive() or plant.mPlantType != PlantType.SPIKEROCK:
                continue
            
            targets = SpikerockCollision.get_zombies_on_tile(plant, self.zombies)
            
            for zombie in targets:
                if zombie.mZombieType in (ZombieType.ZOMBONI, ZombieType.CATAPULT):
                    # Vehicles handled separately
                    continue
                
                zombie.take_damage(plant.mAttackDamage)
    
    # =========================================================================
    # LAWN MOWER
    # =========================================================================
    
    def _update_lawn_mowers(self) -> None:
        """Update lawn mower positions and check triggers."""
        for mower in self.lawn_mowers:
            if not mower.mIsActive:
                continue
            
            if mower.mIsTriggered:
                # Move right and kill zombies
                mower.mX += mower.mSpeed
                
                # Kill zombies in path
                for zombie in self.zombies:
                    if not zombie.is_alive():
                        continue
                    if zombie.mRow != mower.mRow:
                        continue
                    
                    # Check if zombie is hit by mower
                    if mower.mX - 20 <= zombie.mX <= mower.mX + 60:
                        zombie.mHP = 0
                        zombie.mState = ZombieState.DYING
                
                # Remove mower when off screen
                if mower.mX > ZOMBIE_SPAWN_X + 50:
                    mower.mIsActive = False
            else:
                # Check if zombie reached mower
                for zombie in self.zombies:
                    if not zombie.is_alive():
                        continue
                    if zombie.mRow != mower.mRow:
                        continue
                    if zombie.mIsFlying or zombie.mIsDigging:
                        continue
                    
                    if zombie.mX <= mower.mX + 30:
                        mower.mIsTriggered = True
                        break
    
    # =========================================================================
    # CLEANUP AND END CONDITIONS
    # =========================================================================
    
    def _cleanup(self) -> None:
        """Clean up dead entities."""
        # Update dying zombies
        for zombie in self.zombies:
            if zombie.mState == ZombieState.DYING:
                zombie.mStateTimer += 1
                if zombie.mStateTimer >= 50:  # Death animation
                    zombie.mState = ZombieState.DEAD
        
        # Remove dead zombies (keep list for this tick for rendering)
        self.zombies = [z for z in self.zombies if z.mState != ZombieState.DEAD]
        
        # Remove dead plants
        self.plants = [p for p in self.plants if p.is_alive()]
    
    def _check_game_end(self) -> None:
        """Check for win/lose conditions."""
        # Lose condition: zombie reaches left side
        for zombie in self.zombies:
            if not zombie.is_alive():
                continue
            if zombie.mIsDigging or zombie.mIsFlying:
                continue
            
            # Check if zombie passed all lawn mowers
            mower = None
            for m in self.lawn_mowers:
                if m.mRow == zombie.mRow:
                    mower = m
                    break
            
            if mower is None or not mower.mIsActive:
                if zombie.mX <= 0:
                    self.game_over = True
                    self.victory = False
                    return
        
        # Win condition: survived enough waves (for Survival Endless, never really wins)
        # In endless mode, game continues until loss
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_plant_at(self, row: int, col: int) -> Optional[Plant]:
        """Get plant at grid position."""
        if 0 <= row < NUM_ROWS and 0 <= col < NUM_COLS:
            return self._plant_grid[row][col]
        return None
    
    def get_zombies_in_row(self, row: int) -> list[Zombie]:
        """Get all alive zombies in a row."""
        return [z for z in self.zombies if z.is_alive() and z.mRow == row]
    
    def get_all_zombies(self) -> list[Zombie]:
        """Get all alive zombies."""
        return [z for z in self.zombies if z.is_alive()]
    
    def get_all_plants(self) -> list[Plant]:
        """Get all alive plants."""
        return [p for p in self.plants if p.is_alive()]


# =============================================================================
# BATCH OPERATIONS FOR RL
# =============================================================================

def run_simulation(sim: PvZSim, ticks: int) -> list[GameState]:
    """Run simulation for multiple ticks and return states."""
    states = []
    for _ in range(ticks):
        state = sim.step()
        states.append(state)
        if state.game_over:
            break
    return states


def benchmark(num_ticks: int = 100000, seed: int = 42) -> float:
    """
    Benchmark the simulation speed.
    
    Returns:
        FPS (frames/ticks per second)
    """
    import time
    
    sim = PvZSim(seed=seed)
    
    # Plant some plants
    for row in range(NUM_ROWS):
        sim.plant_at(PlantType.SUNFLOWER, row, 0, free=True)
        sim.plant_at(PlantType.TWIN_SUNFLOWER, row, 1, free=True)
        sim.plant_at(PlantType.GLOOM_SHROOM, row, 4, free=True)
        sim.plant_at(PlantType.WINTER_MELON, row, 6, free=True)
    
    # Spawn some zombies
    for _ in range(20):
        row = random.randint(0, NUM_ROWS - 1)
        zombie_type = random.choice([
            ZombieType.ZOMBIE, ZombieType.CONEHEAD, 
            ZombieType.BUCKETHEAD, ZombieType.FOOTBALL
        ])
        sim.spawn_zombie(zombie_type, row)
    
    start_time = time.perf_counter()
    
    for _ in range(num_ticks):
        sim.step()
    
    elapsed = time.perf_counter() - start_time
    fps = num_ticks / elapsed
    
    return fps


if __name__ == "__main__":
    # Run benchmark
    fps = benchmark(100000)
    print(f"Benchmark: {fps:.0f} FPS")
