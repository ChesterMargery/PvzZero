"""
PvZ Simulator Unit Tests
Tests core functionality: collision, state machines, damage system, etc.
"""

import unittest
from consts import (
    ZombieType, ZombieState, PlantType, DamageType,
    ZOMBIE_STATS, CELL_WIDTH, CELL_HEIGHT, GRID_OFFSET_X, GRID_OFFSET_Y,
    NUM_ROWS, NUM_COLS, CHILL_DURATION, FREEZE_DURATION,
)
from state import Rect, Zombie, Plant, StatusEffect
from physics import (
    check_collision, distance, CollisionSystem, 
    FumeAttack, GloomAttack, CherryBombAttack,
)
from engine import PvZSim


class TestRectCollision(unittest.TestCase):
    """Test AABB collision detection."""
    
    def test_overlapping_rects(self):
        """Two overlapping rectangles should collide."""
        rect1 = Rect(0, 0, 50, 50)
        rect2 = Rect(25, 25, 50, 50)
        self.assertTrue(check_collision(rect1, rect2))
    
    def test_non_overlapping_rects(self):
        """Two non-overlapping rectangles should not collide."""
        rect1 = Rect(0, 0, 50, 50)
        rect2 = Rect(100, 100, 50, 50)
        self.assertFalse(check_collision(rect1, rect2))
    
    def test_edge_touching_rects(self):
        """Rectangles touching at edge should not collide (exclusive)."""
        rect1 = Rect(0, 0, 50, 50)
        rect2 = Rect(50, 0, 50, 50)
        self.assertFalse(check_collision(rect1, rect2))
    
    def test_contained_rect(self):
        """A rectangle fully contained in another should collide."""
        rect1 = Rect(0, 0, 100, 100)
        rect2 = Rect(25, 25, 50, 50)
        self.assertTrue(check_collision(rect1, rect2))


class TestStatusEffects(unittest.TestCase):
    """Test zombie status effects."""
    
    def test_chill_effect(self):
        """Chill effect should slow zombie."""
        status = StatusEffect()
        status.apply_chill()
        
        self.assertTrue(status.is_chilled)
        self.assertEqual(status.chill_timer, CHILL_DURATION)
        self.assertEqual(status.get_speed_mult(), 0.5)
        self.assertEqual(status.get_attack_mult(), 0.5)
    
    def test_freeze_effect(self):
        """Freeze effect should stop zombie."""
        status = StatusEffect()
        status.apply_freeze()
        
        self.assertTrue(status.is_frozen)
        self.assertTrue(status.is_immobilized)
        self.assertEqual(status.get_speed_mult(), 0.0)
    
    def test_freeze_clears_chill(self):
        """Freeze should clear chill effect."""
        status = StatusEffect()
        status.apply_chill()
        status.apply_freeze()
        
        self.assertTrue(status.is_frozen)
        self.assertFalse(status.is_chilled)
    
    def test_status_tick_decay(self):
        """Status effects should decay over time."""
        status = StatusEffect()
        status.apply_chill()
        initial_timer = status.chill_timer
        
        for _ in range(100):
            status.tick()
        
        self.assertEqual(status.chill_timer, initial_timer - 100)


class TestZombieCreation(unittest.TestCase):
    """Test zombie creation and stats."""
    
    def test_basic_zombie_stats(self):
        """Basic zombie should have correct stats."""
        zombie = Zombie.create(1, ZombieType.ZOMBIE, 0, 800, 100)
        
        self.assertEqual(zombie.mZombieId, 1)
        self.assertEqual(zombie.mZombieType, ZombieType.ZOMBIE)
        self.assertEqual(zombie.mHP, 270)
        self.assertEqual(zombie.mAccessoryHP, 0)
        self.assertEqual(zombie.mState, ZombieState.WALK)
    
    def test_buckethead_stats(self):
        """Buckethead should have accessory HP."""
        zombie = Zombie.create(2, ZombieType.BUCKETHEAD, 0, 800, 100)
        
        self.assertEqual(zombie.mHP, 270)
        self.assertEqual(zombie.mAccessoryHP, 1100)
        self.assertEqual(zombie.total_hp, 1370)
    
    def test_gargantuar_stats(self):
        """Gargantuar should have high HP."""
        zombie = Zombie.create(3, ZombieType.GARGANTUAR, 0, 800, 100)
        
        self.assertEqual(zombie.mHP, 3000)
        self.assertFalse(zombie.mHasThrownImp)
    
    def test_balloon_initial_state(self):
        """Balloon zombie should start flying."""
        zombie = Zombie.create(4, ZombieType.BALLOON, 0, 800, 100)
        
        self.assertEqual(zombie.mState, ZombieState.FLYING)
        self.assertTrue(zombie.mIsFlying)


class TestZombieDamage(unittest.TestCase):
    """Test zombie damage system."""
    
    def test_normal_damage(self):
        """Normal damage should reduce HP."""
        zombie = Zombie.create(1, ZombieType.ZOMBIE, 0, 800, 100)
        initial_hp = zombie.mHP
        
        zombie.take_damage(100, DamageType.NORMAL)
        
        self.assertEqual(zombie.mHP, initial_hp - 100)
    
    def test_accessory_absorbs_damage(self):
        """Accessory should absorb damage first."""
        zombie = Zombie.create(1, ZombieType.BUCKETHEAD, 0, 800, 100)
        initial_hp = zombie.mHP
        initial_accessory = zombie.mAccessoryHP
        
        zombie.take_damage(100, DamageType.NORMAL)
        
        self.assertEqual(zombie.mHP, initial_hp)  # Base HP unchanged
        self.assertEqual(zombie.mAccessoryHP, initial_accessory - 100)
    
    def test_pierce_bypasses_accessory(self):
        """Pierce damage should bypass accessory."""
        zombie = Zombie.create(1, ZombieType.BUCKETHEAD, 0, 800, 100)
        initial_hp = zombie.mHP
        initial_accessory = zombie.mAccessoryHP
        
        zombie.take_damage(100, DamageType.PIERCE)
        
        self.assertEqual(zombie.mHP, initial_hp - 100)  # Base HP reduced
        self.assertEqual(zombie.mAccessoryHP, initial_accessory)  # Accessory unchanged
    
    def test_freeze_damage_applies_chill(self):
        """Freeze damage should apply chill effect."""
        zombie = Zombie.create(1, ZombieType.ZOMBIE, 0, 800, 100)
        
        zombie.take_damage(20, DamageType.FREEZE)
        
        self.assertTrue(zombie.mStatus.is_chilled)
    
    def test_lethal_damage_triggers_death(self):
        """Lethal damage should trigger death state."""
        zombie = Zombie.create(1, ZombieType.ZOMBIE, 0, 800, 100)
        
        died = zombie.take_damage(9999, DamageType.NORMAL)
        
        self.assertTrue(died)
        self.assertEqual(zombie.mState, ZombieState.DYING)


class TestPlantCreation(unittest.TestCase):
    """Test plant creation and stats."""
    
    def test_sunflower_creation(self):
        """Sunflower should be created with correct stats."""
        plant = Plant.create(1, PlantType.SUNFLOWER, 0, 0)
        
        self.assertEqual(plant.mPlantId, 1)
        self.assertEqual(plant.mPlantType, PlantType.SUNFLOWER)
        self.assertEqual(plant.mHP, 300)
        self.assertEqual(plant.mRow, 0)
        self.assertEqual(plant.mCol, 0)
    
    def test_pumpkin_hp(self):
        """Pumpkin should have high HP."""
        plant = Plant.create(2, PlantType.PUMPKIN, 0, 0)
        
        self.assertEqual(plant.mHP, 4000)
    
    def test_mushroom_starts_asleep(self):
        """Mushrooms should start asleep."""
        fume = Plant.create(3, PlantType.FUME_SHROOM, 0, 0)
        gloom = Plant.create(4, PlantType.GLOOM_SHROOM, 1, 0)
        
        self.assertFalse(fume.mIsAwake)
        self.assertFalse(gloom.mIsAwake)
    
    def test_pumpkin_layer(self):
        """Adding pumpkin should add HP layer."""
        plant = Plant.create(5, PlantType.SUNFLOWER, 0, 0)
        plant.add_pumpkin()
        
        self.assertTrue(plant.mHasPumpkin)
        self.assertEqual(plant.mPumpkinHP, 4000)


class TestSimulator(unittest.TestCase):
    """Test main simulator functionality."""
    
    def test_initialization(self):
        """Simulator should initialize with correct defaults."""
        sim = PvZSim(seed=42)
        
        self.assertEqual(sim.tick, 0)
        self.assertEqual(sim.sun, 50)
        self.assertEqual(len(sim.zombies), 0)
        self.assertEqual(len(sim.plants), 0)
        self.assertFalse(sim.game_over)
    
    def test_plant_placement(self):
        """Plants should be placed correctly."""
        sim = PvZSim(seed=42)
        
        plant = sim.plant_at(PlantType.SUNFLOWER, 0, 0, free=True)
        
        self.assertIsNotNone(plant)
        self.assertEqual(plant.mRow, 0)
        self.assertEqual(plant.mCol, 0)
        self.assertEqual(len(sim.plants), 1)
    
    def test_plant_costs_sun(self):
        """Planting should cost sun."""
        sim = PvZSim(seed=42)
        sim.sun = 100
        
        plant = sim.plant_at(PlantType.SUNFLOWER, 0, 0, free=False)
        
        self.assertIsNotNone(plant)
        self.assertEqual(sim.sun, 50)  # Sunflower costs 50
    
    def test_insufficient_sun(self):
        """Can't plant without enough sun."""
        sim = PvZSim(seed=42)
        sim.sun = 0
        
        plant = sim.plant_at(PlantType.SUNFLOWER, 0, 0, free=False)
        
        self.assertIsNone(plant)
    
    def test_zombie_spawning(self):
        """Zombies should spawn correctly."""
        sim = PvZSim(seed=42)
        
        zombie = sim.spawn_zombie(ZombieType.ZOMBIE, 0)
        
        self.assertIsNotNone(zombie)
        self.assertEqual(zombie.mRow, 0)
        self.assertEqual(zombie.mX, 800)
        self.assertEqual(len(sim.zombies), 1)
    
    def test_step_advances_tick(self):
        """Step should advance tick counter."""
        sim = PvZSim(seed=42)
        
        sim.step()
        
        self.assertEqual(sim.tick, 1)
    
    def test_zombie_movement(self):
        """Zombies should move towards house."""
        sim = PvZSim(seed=42)
        zombie = sim.spawn_zombie(ZombieType.ZOMBIE, 0)
        initial_x = zombie.mX
        
        for _ in range(100):
            sim.step()
        
        self.assertLess(zombie.mX, initial_x)
    
    def test_sunflower_produces_sun(self):
        """Sunflower should produce sun over time."""
        sim = PvZSim(seed=42)
        sim.plant_at(PlantType.SUNFLOWER, 0, 0, free=True)
        initial_sun = sim.sun
        
        # Run for full sun production cycle (2400 ticks)
        for _ in range(2500):
            sim.step()
        
        self.assertGreater(sim.sun, initial_sun)
    
    def test_zombie_eating_plant(self):
        """Zombie should eat and damage plant."""
        sim = PvZSim(seed=42)
        plant = sim.plant_at(PlantType.SUNFLOWER, 0, 0, free=True)
        zombie = sim.spawn_zombie(ZombieType.ZOMBIE, 0, x=100)  # Close to plant
        initial_hp = plant.mHP
        
        # Run until zombie reaches plant and eats
        for _ in range(1000):
            sim.step()
            if plant.mHP < initial_hp:
                break
        
        self.assertLess(plant.mHP, initial_hp)
    
    def test_game_reset(self):
        """Reset should clear all state."""
        sim = PvZSim(seed=42)
        sim.plant_at(PlantType.SUNFLOWER, 0, 0, free=True)
        sim.spawn_zombie(ZombieType.ZOMBIE, 0)
        sim.tick = 1000
        sim.sun = 500
        
        sim.reset()
        
        self.assertEqual(sim.tick, 0)
        self.assertEqual(sim.sun, 50)
        self.assertEqual(len(sim.zombies), 0)
        self.assertEqual(len(sim.plants), 0)


class TestCollisionSystem(unittest.TestCase):
    """Test collision system helpers."""
    
    def test_get_zombie_plant_collisions(self):
        """Should detect zombie-plant collisions."""
        sim = PvZSim(seed=42)
        plant = sim.plant_at(PlantType.SUNFLOWER, 0, 0, free=True)
        zombie = sim.spawn_zombie(ZombieType.ZOMBIE, 0, x=60)  # On top of plant
        
        collisions = CollisionSystem.get_zombie_plant_collisions(
            sim.zombies, sim.plants
        )
        
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0][0], zombie)
        self.assertEqual(collisions[0][1], plant)
    
    def test_no_collision_different_rows(self):
        """No collision if zombie and plant in different rows."""
        sim = PvZSim(seed=42)
        plant = sim.plant_at(PlantType.SUNFLOWER, 0, 0, free=True)
        zombie = sim.spawn_zombie(ZombieType.ZOMBIE, 1, x=60)  # Different row
        
        collisions = CollisionSystem.get_zombie_plant_collisions(
            sim.zombies, sim.plants
        )
        
        self.assertEqual(len(collisions), 0)


class TestAttackPatterns(unittest.TestCase):
    """Test plant attack patterns."""
    
    def test_fume_attack_range(self):
        """Fume-shroom should hit zombies in line."""
        sim = PvZSim(seed=42)
        plant = sim.plant_at(PlantType.FUME_SHROOM, 0, 0, free=True)
        plant.mIsAwake = True
        
        # Zombie in range
        zombie1 = sim.spawn_zombie(ZombieType.ZOMBIE, 0, x=200)
        # Zombie out of range  
        zombie2 = sim.spawn_zombie(ZombieType.ZOMBIE, 0, x=500)
        
        targets = FumeAttack.get_targets(plant, sim.zombies)
        
        self.assertIn(zombie1, targets)
        self.assertNotIn(zombie2, targets)
    
    def test_gloom_attack_aoe(self):
        """Gloom-shroom should hit zombies in 3x3."""
        sim = PvZSim(seed=42)
        plant = sim.plant_at(PlantType.GLOOM_SHROOM, 1, 4, free=True)
        plant.mIsAwake = True
        
        # Zombie in adjacent row (should hit)
        zombie1 = sim.spawn_zombie(ZombieType.ZOMBIE, 0, x=plant.mX + 50)
        # Zombie same row (should hit)
        zombie2 = sim.spawn_zombie(ZombieType.ZOMBIE, 1, x=plant.mX - 30)
        
        targets = GloomAttack.get_targets(plant, sim.zombies)
        
        self.assertIn(zombie1, targets)
        self.assertIn(zombie2, targets)
    
    def test_cherry_bomb_explosion(self):
        """Cherry Bomb should hit all zombies in radius."""
        sim = PvZSim(seed=42)
        # Use row 0 (not pool) to avoid lily pad requirement
        plant = sim.plant_at(PlantType.CHERRY_BOMB, 0, 4, free=True)
        
        # Zombie in explosion radius
        zombie1 = sim.spawn_zombie(ZombieType.ZOMBIE, 0, x=plant.mX + 50)
        # Zombie in adjacent row but close
        zombie2 = sim.spawn_zombie(ZombieType.ZOMBIE, 1, x=plant.mX)
        # Zombie far away
        zombie3 = sim.spawn_zombie(ZombieType.ZOMBIE, 0, x=plant.mX + 300)
        
        targets = CherryBombAttack.get_targets(plant, sim.zombies)
        
        self.assertIn(zombie1, targets)
        self.assertIn(zombie2, targets)
        self.assertNotIn(zombie3, targets)


class TestSpecialZombies(unittest.TestCase):
    """Test special zombie mechanics."""
    
    def test_pole_vaulter_jumps(self):
        """Pole vaulter should jump over first plant."""
        sim = PvZSim(seed=42)
        plant = sim.plant_at(PlantType.SUNFLOWER, 0, 7, free=True)
        zombie = sim.spawn_zombie(ZombieType.POLE_VAULTER, 0)
        
        # Run until zombie encounters plant
        for _ in range(2000):
            sim.step()
            if zombie.mHasJumped:
                break
        
        self.assertTrue(zombie.mHasJumped)
    
    def test_balloon_is_flying(self):
        """Balloon zombie should be flying initially."""
        sim = PvZSim(seed=42)
        zombie = sim.spawn_zombie(ZombieType.BALLOON, 0)
        
        self.assertTrue(zombie.mIsFlying)
        self.assertFalse(zombie.is_ground_targetable())
    
    def test_snorkel_is_submerged(self):
        """Snorkel zombie should start submerged."""
        sim = PvZSim(seed=42)
        zombie = sim.spawn_zombie(ZombieType.SNORKEL, 2)  # Pool row
        
        self.assertTrue(zombie.mIsSubmerged)
        self.assertFalse(zombie.is_targetable())


class TestLawnMowers(unittest.TestCase):
    """Test lawn mower mechanics."""
    
    def test_lawn_mowers_initialized(self):
        """Each row should have a lawn mower."""
        sim = PvZSim(seed=42)
        
        self.assertEqual(len(sim.lawn_mowers), NUM_ROWS)
        for i, mower in enumerate(sim.lawn_mowers):
            self.assertEqual(mower.mRow, i)
            self.assertTrue(mower.mIsActive)
    
    def test_lawn_mower_triggers(self):
        """Lawn mower should trigger when zombie reaches it."""
        sim = PvZSim(seed=42)
        zombie = sim.spawn_zombie(ZombieType.ZOMBIE, 0, x=20)  # Near house
        mower = sim.lawn_mowers[0]
        
        # Run until mower triggers
        for _ in range(100):
            sim.step()
            if mower.mIsTriggered:
                break
        
        self.assertTrue(mower.mIsTriggered)


class TestPerformance(unittest.TestCase):
    """Test simulation performance."""
    
    def test_simulation_speed(self):
        """Simulation should run fast enough for RL."""
        import time
        
        sim = PvZSim(seed=42)
        
        # Set up a reasonable game state
        for row in range(NUM_ROWS):
            sim.plant_at(PlantType.SUNFLOWER, row, 0, free=True)
            sim.plant_at(PlantType.GLOOM_SHROOM, row, 4, free=True)
        
        for row in range(NUM_ROWS):
            sim.spawn_zombie(ZombieType.ZOMBIE, row)
            sim.spawn_zombie(ZombieType.CONEHEAD, row)
        
        num_ticks = 10000
        start = time.perf_counter()
        
        for _ in range(num_ticks):
            sim.step()
        
        elapsed = time.perf_counter() - start
        fps = num_ticks / elapsed
        
        # Should exceed 10,000 FPS target
        self.assertGreater(fps, 10000, f"FPS {fps:.0f} below target 10,000")


if __name__ == "__main__":
    unittest.main()
