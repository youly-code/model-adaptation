import numpy as np
import pygame
from typing import Tuple, List
from dataclasses import dataclass
import time

@dataclass
class SimulationConfig:
    """Configuration parameters for the swarm simulation."""
    num_boids: int = 300
    width: int = 1600
    height: int = 1000
    max_speed: float = 8.0
    min_speed: float = 4.0
    perception_radius: float = 150.0
    alignment_weight: float = 1.5     # Increased for better flocking
    cohesion_weight: float = 0.01     # Increased but still gentle
    separation_weight: float = 1.2
    boid_size: int = 4
    turn_speed: float = 0.15
    min_separation: float = 45.0
    max_force: float = 0.4
    draw_perception: bool = False
    global_speed: float = 0.02    # Reduced global influence
    wind_strength: float = 12.0    # Increased dramatically
    wind_turbulence: float = 1.2   # More turbulence
    wind_duration: float = 3.0     # Longer wind effect

class Boids:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.positions = np.random.rand(config.num_boids, 2) * [config.width, config.height]
        angles = np.random.uniform(0, 2*np.pi, config.num_boids)
        self.velocities = np.array([
            np.cos(angles),
            np.sin(angles)
        ]).T * np.random.uniform(config.min_speed, config.max_speed, config.num_boids)[:, np.newaxis]
        self.last_event_time = time.time()
        self.event_interval = np.random.uniform(5, 10)
        self.current_direction = np.array([1.0, 0.0])
        self.disturbance_duration = 1.0
        self.disturbance_center = np.mean(self.positions, axis=0)
        self.wind_direction = np.array([0.0, 0.0])
        self.wind_change_time = time.time()
        self.wind_duration = config.wind_duration
        self.wind_phase = 0
        self.last_wind_shift = time.time()
        self.wind_shift_interval = 0.5  # Time between direction changes during gust

    def update(self) -> None:
        current_time = time.time()
        
        # Wind event
        if current_time - self.last_event_time > self.event_interval:
            # Stronger initial gust
            wind_angle = np.random.uniform(0, 2*np.pi)
            self.wind_direction = np.array([np.cos(wind_angle), np.sin(wind_angle)])
            self.wind_change_time = current_time
            self.last_event_time = current_time
            self.last_wind_shift = current_time
            self.event_interval = np.random.uniform(5, 10)
            self.disturbance_center = np.mean(self.positions, axis=0)
            
            # Massive initial push
            self.velocities += self.wind_direction * self.config.wind_strength * 6.0
        
        # Calculate wind influence
        wind_factor = max(0, 1.0 - (current_time - self.wind_change_time) / self.wind_duration)
        
        if wind_factor > 0:
            # Sudden direction changes during the gust
            if current_time - self.last_wind_shift > self.wind_shift_interval:
                # Dramatic direction change
                shift_angle = np.random.uniform(-np.pi/2, np.pi/2)  # 90-degree range
                cos_shift = np.cos(shift_angle)
                sin_shift = np.sin(shift_angle)
                
                # Rotate wind direction
                new_x = self.wind_direction[0] * cos_shift - self.wind_direction[1] * sin_shift
                new_y = self.wind_direction[0] * sin_shift + self.wind_direction[1] * cos_shift
                self.wind_direction = np.array([new_x, new_y])
                
                # Add extra burst of force on direction change
                self.velocities += self.wind_direction * self.config.wind_strength * wind_factor * 2.0
                
                self.last_wind_shift = current_time
            
            # More dramatic distance-based effect
            distances_to_center = np.linalg.norm(self.positions - self.disturbance_center, axis=1)
            wind_influence = np.exp(-distances_to_center / (self.config.perception_radius * 4))
            
            # Stronger turbulence with vertical bias
            turbulence = np.random.randn(self.config.num_boids, 2) * self.config.wind_turbulence
            turbulence[:, 1] *= 1.5  # Extra vertical chaos
            
            # Base wind force with turbulence
            wind_force = (self.wind_direction + turbulence) * self.config.wind_strength * wind_factor
            
            # Add swirling effect
            time_phase = current_time * 2.0
            swirl = np.array([
                np.cos(time_phase) * wind_factor,
                np.sin(time_phase) * wind_factor
            ]) * self.config.wind_strength * 0.5
            
            wind_force += swirl
            
            # Apply forces
            self.velocities += wind_force * wind_influence[:, np.newaxis]
            
            # Much stronger separation during intense wind
            self.config.separation_weight = 1.2 * (1 + wind_factor * 4.0)

        # Normal flocking behavior
        for i in range(self.config.num_boids):
            distances = np.linalg.norm(self.positions - self.positions[i], axis=1)
            neighbors = distances < self.config.perception_radius
            neighbors[i] = False
            
            if neighbors.sum() > 0:
                weights = 1 / (distances[neighbors] + 1e-6)
                
                # Stronger alignment during wind to maintain formation
                alignment = np.average(self.velocities[neighbors], axis=0, weights=weights)
                alignment_force = (alignment - self.velocities[i]) * (
                    self.config.alignment_weight * (1.0 + wind_factor * 0.5)
                )
                
                # Maintain cohesion against wind
                cohesion = np.average(self.positions[neighbors], axis=0, weights=weights) - self.positions[i]
                cohesion_force = cohesion * self.config.cohesion_weight
                
                # Separation
                separation = np.sum(
                    (self.positions[i] - self.positions[neighbors]) / 
                    (distances[neighbors, np.newaxis]**2 + 1e-6), axis=0
                )
                separation_force = separation * self.config.separation_weight
                
                self.velocities[i] += alignment_force + cohesion_force + separation_force

        # Normalize velocities after all forces are applied
        speeds = np.linalg.norm(self.velocities, axis=1)
        too_fast = speeds > self.config.max_speed
        if np.any(too_fast):
            self.velocities[too_fast] *= (self.config.max_speed / speeds[too_fast, np.newaxis])
        
        too_slow = speeds < self.config.min_speed
        if np.any(too_slow):
            self.velocities[too_slow] *= (self.config.min_speed / speeds[too_slow, np.newaxis])

        # Add current direction with reduced influence
        global_direction = self.current_direction * self.config.global_speed * (1.0 - np.mean(wind_factor))
        self.velocities += global_direction
        
        # Final velocity normalization
        self.velocities = np.clip(
            self.velocities, 
            -self.config.max_speed, 
            self.config.max_speed
        )
        
        # Update positions
        self.positions += self.velocities
        self.positions %= [self.config.width, self.config.height]

def rotate_point(point: Tuple[float, float], angle: float) -> np.ndarray:
    """Rotate a point around origin by given angle in radians."""
    x, y = point
    return np.array([
        x * np.cos(angle) - y * np.sin(angle),
        x * np.sin(angle) + y * np.cos(angle)
    ])

def main():
    config = SimulationConfig()
    pygame.init()
    screen = pygame.display.set_mode((config.width, config.height))
    clock = pygame.time.Clock()
    boids = Boids(config)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # In the main loop
        mouse_pos = np.array(pygame.mouse.get_pos())
        if pygame.mouse.get_pressed()[0]:  # Left mouse button
            # Create attraction to mouse
            for i in range(boids.config.num_boids):
                diff = mouse_pos - boids.positions[i]
                dist = np.linalg.norm(diff)
                if dist < 200:  # Mouse influence radius
                    boids.velocities[i] += diff * 0.001  # Attraction
        elif pygame.mouse.get_pressed()[2]:  # Right mouse button
            # Create repulsion from mouse
            for i in range(boids.config.num_boids):
                diff = mouse_pos - boids.positions[i]
                dist = np.linalg.norm(diff)
                if dist < 200:  # Mouse influence radius
                    boids.velocities[i] -= diff * 0.002  # Stronger repulsion
        
        screen.fill((0, 0, 0))  # Black background
        
        # Update and draw boids
        boids.update()
        for pos, vel in zip(boids.positions, boids.velocities):
            # Calculate heading angle
            angle = np.arctan2(vel[1], vel[0])
            # Triangle points
            points = [
                pos + rotate_point(( 0,  8), angle),
                pos + rotate_point((-4, -4), angle),
                pos + rotate_point(( 4, -4), angle)
            ]
            pygame.draw.polygon(screen, (255, 255, 255), points)
            
            # Optionally draw perception radius
            if config.draw_perception:
                pygame.draw.circle(
                    screen, (50, 50, 50), pos.astype(int),
                    int(config.perception_radius), 1
                )
        
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    pygame.quit()

if __name__ == "__main__":
    main()
