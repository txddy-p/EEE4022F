import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
from enum import Enum
import random
import time
import numpy as np

# Define map types
class MapType(Enum):
    STRAIGHT = 0
    T_JUNCTION = 1
    NARROW = 2
    PHYSICS = 3

# Robot class definition
class Robot:
    def __init__(self, radius=0.087, wheel_diameter=0.066, motor_constant=0.0069, 
                 max_voltage=6.0, max_sensor_range=2):
        # Physical specifications
        self.radius = radius  # 18cm diameter = 8.7cm radius
        self.wheel_diameter = wheel_diameter  # 66mm = 0.066m
        self.wheel_radius = wheel_diameter / 2
        self.wheel_circumference = math.pi * wheel_diameter  # ~0.207m
        self.wheel_distance = 0.11  # Distance between wheels (robot diameter)
        
        # Motor parameters
        self.motor_constant = motor_constant  # Motor constant Kv (rad/s/V)
        self.max_voltage = max_voltage  # Maximum voltage (6V)
        self.reduction_ratio = 48  # Gear ratio 48:1
        
        # Derived maximum velocities based on no-load speed: 240 RPM
        self.max_rpm = 240  # Max 240 RPM
        self.max_wheel_speed = (self.max_rpm / 60) * self.wheel_circumference  # ~0.828 m/s
        
        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # Heading angle in radians
        self.v = 0.0  # Current linear velocity
        self.w = 0.0  # Current angular velocity
        self.left_voltage = 0.0  # Left wheel voltage
        self.right_voltage = 0.0  # Right wheel voltage
        
        # Sensors
        self.max_sensor_range = max_sensor_range
        self.left_sensor = max_sensor_range
        self.front_sensor = max_sensor_range
        self.right_sensor = max_sensor_range
        self.color_sensor = 0  # 0 = not at goal, 1 = at goal
        
        # Sensor angles relative to robot heading
        self.left_sensor_angle = math.pi/4  # 45 degrees to the left
        self.front_sensor_angle = 0  # Straight ahead
        self.right_sensor_angle = -math.pi/4  # 45 degrees to the right
    
    def reset(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0
        self.w = 0.0
        self.left_voltage = 0.0
        self.right_voltage = 0.0
        self.left_sensor = self.max_sensor_range
        self.front_sensor = self.max_sensor_range
        self.right_sensor = self.max_sensor_range
        self.color_sensor = 0
    
    def update(self, left_voltage, right_voltage, dt, walls, goal):
        # Store voltages
        self.left_voltage = np.clip(left_voltage, -self.max_voltage, self.max_voltage)
        self.right_voltage = np.clip(right_voltage, -self.max_voltage, self.max_voltage)
        
        # Convert voltages to wheel angular velocities (rad/s)
        left_angular_vel = (self.left_voltage / self.motor_constant) / self.reduction_ratio
        right_angular_vel = (self.right_voltage / self.motor_constant) / self.reduction_ratio
        
        # Convert angular velocities to linear wheel speeds (m/s)
        v_left = left_angular_vel * self.wheel_radius
        v_right = right_angular_vel * self.wheel_radius
        
        # Apply differential drive kinematics
        self.v = (v_right + v_left) / 2.0  # Linear velocity
        self.w = (v_right - v_left) / self.wheel_distance  # Angular velocity
        
        # Update orientation
        self.theta += self.w * dt
        self.theta = self.normalize_angle(self.theta)
        
        # Calculate new position
        dx = self.v * math.cos(self.theta) * dt
        dy = self.v * math.sin(self.theta) * dt
        
        # Store old position
        old_x, old_y = self.x, self.y
        
        # Update position
        self.x += dx
        self.y += dy
        
        # Check for collision
        if self.check_collision(walls):
            # If collision would occur, revert position and stop robot
            self.x, self.y = old_x, old_y
            self.v = 0.0
            self.w = 0.0
            return True  # Collision occurred
        
        # Update sensors
        self.update_sensors(walls, goal)
        
        return False  # No collision
    
    def check_collision(self, walls):
        # Check if robot's circle intersects with any wall
        for wall in walls:
            x1, y1, x2, y2 = wall
            
            # Calculate closest point on line segment to robot center
            line_vec = np.array([x2 - x1, y2 - y1])
            line_len = np.linalg.norm(line_vec)
            line_unit_vec = line_vec / line_len if line_len > 0 else np.array([0, 0])
            robot_vec = np.array([self.x - x1, self.y - y1])  # Vector from line start to robot
            projection = np.dot(robot_vec, line_unit_vec)  # Projection of robot_vec onto line_unit_vec
            projection = np.clip(projection, 0, line_len)  # Clamp projection to line segment
            closest_point = np.array([x1, y1]) + projection * line_unit_vec  # Closest point on line segment to robot
            distance = np.linalg.norm(np.array([self.x, self.y]) - closest_point)  # Distance from robot to closest point
            
            # Check if robot is colliding with the wall
            if distance < self.radius:
                return True
        
        return False
    
    def update_sensors(self, walls, goal):
        # Reset sensors to maximum range
        self.left_sensor = self.max_sensor_range
        self.front_sensor = self.max_sensor_range
        self.right_sensor = self.max_sensor_range
        
        # Ray tracing for each sensor
        sensor_angles = [self.left_sensor_angle, self.front_sensor_angle, self.right_sensor_angle]
        for i, angle in enumerate(sensor_angles):
            # Calculate sensor position on the circumference of the robot
            sensor_angle = self.theta + angle
            
            # Position the sensor on the circumference of the robot
            sensor_x = self.x + self.radius * math.cos(sensor_angle)
            sensor_y = self.y + self.radius * math.sin(sensor_angle)
            
            # Cast ray from the sensor position
            sensor_value = self.ray_cast(sensor_x, sensor_y, sensor_angle, walls)
            
            # After calculating sensor_value
            sensor_noise_std = 0.01
            sensor_value_noisy = sensor_value + np.random.normal(0, sensor_noise_std)
            
            # Update the appropriate sensor
            if i == 0:
                self.left_sensor = sensor_value_noisy
            elif i == 1:
                self.front_sensor = sensor_value_noisy
            elif i == 2:
                self.right_sensor = sensor_value_noisy
        
        # Check color sensor
        self.color_sensor = 1 if self.is_at_goal(goal) else 0
    
    def ray_cast(self, x, y, angle, walls):
        # Cast a ray from (x, y) in direction angle and return distance to nearest wall
        ray_dir = np.array([math.cos(angle), math.sin(angle)])
        min_distance = self.max_sensor_range
        
        for wall in walls:
            x1, y1, x2, y2 = wall
            
            # Convert wall to vector form
            wall_start = np.array([x1, y1])
            wall_vec = np.array([x2 - x1, y2 - y1])
            ray_start = np.array([x, y])  # Ray starting point (on robot circumference)
            
            # Ray-line segment intersection calculation
            cross_product = ray_dir[0] * wall_vec[1] - ray_dir[1] * wall_vec[0]  # Calculate determinant
            
            # If rays are parallel, no intersection
            if abs(cross_product) < 1e-10:
                continue
            
            # Calculate intersection parameters
            t = ((wall_start[0] - ray_start[0]) * wall_vec[1] - (wall_start[1] - ray_start[1]) * wall_vec[0]) / cross_product
            u = ((wall_start[0] - ray_start[0]) * ray_dir[1] - (wall_start[1] - ray_start[1]) * ray_dir[0]) / cross_product
            
            # Check if intersection is valid (on ray and on wall segment)
            if t >= 0 and 0 <= u <= 1:
                distance = t
                if distance < min_distance:
                    min_distance = distance
        
        return min_distance
    

    def is_at_goal(self, goal):
        # The color sensor is at the front of the robot (on the circumference)
        sensor_x = self.x + self.radius * math.cos(self.theta)
        sensor_y = self.y + self.radius * math.sin(self.theta)
        goal_x, goal_y, goal_width, goal_height = goal
        half_w = goal_width / 2
        half_h = goal_height / 2
        # Check if the color sensor is inside the goal rectangle
        if (goal_x - half_w <= sensor_x <= goal_x + half_w and
            goal_y - half_h <= sensor_y <= goal_y + half_h):
            return True
        return False

    
    def get_state(self):
        # Return the robot's current state (sensor readings + previous velocities)
        return np.array([
            self.left_sensor, 
            self.front_sensor, 
            self.right_sensor, 
            self.color_sensor,
            self.v, 
            self.w
        ])
    
    def normalize_angle(self, angle):
        return ((angle + math.pi) % (2 * math.pi)) - math.pi  # Normalize angle to be between -pi and pi

# Robot Environment
class RobotEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, map_type=MapType.STRAIGHT, render_mode='human'):
        super(RobotEnv, self).__init__()
        
        # Environment settings
        self.map_type = map_type
        self.dt = 0.1  # Time step
        self.max_steps = 1000  # Maximum steps per episode
        self.step_count = 0
        self.render_mode = render_mode
        
        # Robot
        self.robot = Robot()
        
        # Define action and observation spaces
        # Action: [left_voltage, right_voltage]
        self.action_space = spaces.Box(
            low=np.array([-self.robot.max_voltage, -self.robot.max_voltage]),
            high=np.array([self.robot.max_voltage, self.robot.max_voltage]),
            dtype=np.float32
        )
        
        # Observation: [left_sensor, front_sensor, right_sensor, color_sensor, v, w]
        self.observation_space = spaces.Box(
            low=np.array([0.02, 0.02, 0.02, 0, -self.robot.max_wheel_speed, -self.robot.max_wheel_speed/self.robot.radius]),
            high=np.array([
                self.robot.max_sensor_range,
                self.robot.max_sensor_range,
                self.robot.max_sensor_range,
                1,
                self.robot.max_wheel_speed,
                self.robot.max_wheel_speed/self.robot.radius
            ]),
            dtype=np.float32
        )
        
        # Initialize maps and robot position
        self.walls = []
        self.goal = None
        self.create_map()
        
        # Initialize Pygame for visualization
        self.screen = None
        self.clock = None
        self.isopen = False
    
    def create_map(self):
        if self.map_type == MapType.STRAIGHT:
            # Straight line from left to right
            self.map_width = 10 * self.robot.radius  # 1.8 meters (10 * 0.18m)
            self.map_height = 4 * self.robot.radius  # 0.36 meters (2 * 0.18m)
            
            # Define walls (rectangular corridor)
            self.walls = [
                (0.0, 0.0, self.map_width, 0.0),                           # Bottom wall
                (0.0, self.map_height, self.map_width, self.map_height),   # Top wall
                (0.0, 0.0, 0.0, self.map_height),                          # Left wall
                (self.map_width, 0.0, self.map_width, self.map_height)     # Right wall
            ]
           
            
            # Define goal as a rectangle
            goal_x = self.map_width - self.robot.radius / 2  # Center of goal square
            goal_y = self.map_height / 2

            goal_width = self.robot.radius * 1.5              # Rectangle width
            goal_height = self.map_height                  # Rectangle height (example)
            self.goal = (goal_x, goal_y, goal_width, goal_height)
            
            # Define start position (one robot radius from left wall)
            self.start_x = self.robot.radius + 0.03
            self.start_y = self.map_height / 2

        elif self.map_type == MapType.NARROW:
            # Narrow corridor with dimensions proportional to robot radius
            self.map_width = 12 * self.robot.radius  # (12 * 0.18m)
            self.map_height = 5 * self.robot.radius  # (5 * 0.18m)
            
            self.walls = [
                (0.0, 0.0, self.map_width, 1.25* self.robot.radius),                           # Bottom wall
                (0.0, self.map_height, self.map_width, 3.75*self.robot.radius),   # Top wall
                (0.0, 0.0, 0.0, self.map_height),                          # Left wall
                (self.map_width, 1.25 * self.robot.radius, self.map_width, 3.75*self.robot.radius)     # Right wall
            ]
            
            

            # Define goal as a rectangle
            goal_x = self.map_width - self.robot.radius / 2  # Center of goal square
            goal_y = self.map_height / 2

            goal_width = self.robot.radius * 1.5             # Rectangle width
            goal_height = 2.5 * self.robot.radius                  # Rectangle height (example)
            self.goal = (goal_x, goal_y, goal_width, goal_height)
            # Define start position (one robot radius from left wall)
            self.start_x = self.robot.radius + 0.03
            self.start_y = self.map_height / 2
            
        elif self.map_type == MapType.T_JUNCTION:
            # T-junction with dimensions proportional to robot radius
            self.map_width = 12 * self.robot.radius  # 1.8 meters with 18cm robot
            self.map_height = 12 * self.robot.radius  # Same as width for square map
            
            # Define corridor width (2 robot diameters)
            corridor_width = 2 * self.robot.radius * 2  # 2 * robot diameter
            
            # Calculate positions
            v_left_x = self.map_width / 2 - corridor_width / 2
            v_right_x = self.map_width / 2 + corridor_width / 2
            v_bottom_y = 0
            v_top_y = self.map_height / 2 - corridor_width / 2
            
            h_left_x = 0
            h_right_x = self.map_width
            h_top_y = self.map_height / 2 + corridor_width / 2
            h_bottom_y = self.map_height / 2 - corridor_width / 2
            
            # Define walls for T-junction
            self.walls = [
                # Vertical corridor walls
                (v_left_x, v_bottom_y, v_left_x, v_top_y),      # Left wall
                (v_right_x, v_bottom_y, v_right_x, v_top_y),    # Right wall
                
                # Horizontal corridor walls
                (h_left_x, h_bottom_y, v_left_x, h_bottom_y),   # Bottom-left segment
                (v_right_x, h_bottom_y, h_right_x, h_bottom_y), # Bottom-right segment
                (h_left_x, h_top_y, h_right_x, h_top_y),        # Top wall
                
                # End caps
                (h_left_x, h_bottom_y, h_left_x, h_top_y),      # Left end
                (h_right_x, h_bottom_y, h_right_x, h_top_y),    # Right end

                (v_left_x, v_bottom_y, v_right_x, v_bottom_y) # bottom end
            ]
            
            # Define goal (on the right side of the T)
            goal_x = self.map_width - self.robot.radius
            goal_y = self.map_height / 2


            # Define goal as a rectangle
            goal_x = self.map_width - self.robot.radius / 2  # Center of goal square
            goal_y = self.map_height / 2

            goal_width = self.robot.radius * 2              # Rectangle width
            goal_height = 4 * self.robot.radius                  # Rectangle height (example)
            self.goal = (goal_x, goal_y, goal_width, goal_height)
            
            # Define start position (at the bottom of the T)
            self.start_x = self.map_width / 2
            self.start_y = self.robot.radius + 0.01  # One robot radius from bottom
        
        elif self.map_type == MapType.PHYSICS:
            # Large open area for physics/fidelity testing
            self.map_width = 20 * self.robot.radius   # Large width (e.g., 1.74m)
            self.map_height = 20 * self.robot.radius  # Large height (e.g., 1.74m)

            # Define walls as a large rectangle (arena)
            self.walls = [
                (0.0, 0.0, self.map_width, 0.0),                           # Bottom wall
                (0.0, self.map_height, self.map_width, self.map_height),   # Top wall
                (0.0, 0.0, 0.0, self.map_height),                          # Left wall
                (self.map_width, 0.0, self.map_width, self.map_height)     # Right wall
            ]

            # Place a small goal somewhere in the arena (e.g., top right corner)
            goal_x = self.map_width - 2 * self.robot.radius
            goal_y = self.map_height - 2 * self.robot.radius
            goal_width = self.robot.radius * 1.5
            goal_height = self.robot.radius * 1.5
            self.goal = (goal_x, goal_y, goal_width, goal_height)

            # Start position: bottom left corner, one robot radius from each wall
            self.start_x = self.robot.radius + 0.03
            self.start_y = self.robot.radius + 0.03

    def demo_trajectories(self):
        """
        Demonstrate basic robot motions and draw their trajectories:
        - Forward
        - Backward
        - Turn in place (left/right)
        - Arching turn (left/right)
        """
        motions = [
            {"name": "Forward", "voltages": (self.robot.max_voltage, self.robot.max_voltage)},
            {"name": "Backward", "voltages": (-self.robot.max_voltage, -self.robot.max_voltage)},
            {"name": "Turn Left In Place", "voltages": (-self.robot.max_voltage, self.robot.max_voltage)},
            {"name": "Turn Right In Place", "voltages": (self.robot.max_voltage, -self.robot.max_voltage)},
            {"name": "Arch Left", "voltages": (self.robot.max_voltage * 0.5, self.robot.max_voltage)},
            {"name": "Arch Right", "voltages": (self.robot.max_voltage, self.robot.max_voltage * 0.5)},
        ]
        steps_per_motion = 50
        pause_time = 0.8

        for motion in motions:
            # Reset robot to center of map, facing right (theta=0)
            self.robot.reset(self.map_width / 2, self.map_height / 2, 0)
            trajectory = [(self.robot.x, self.robot.y)]
            for _ in range(steps_per_motion):
                self.robot.update(*motion["voltages"], self.dt, self.walls, self.goal)
                trajectory.append((self.robot.x, self.robot.y))
                self.render()
                # Draw trajectory so far
                self.draw_trajectory(trajectory)
                pygame.display.flip()
                self.clock.tick(1 / self.dt)
            # Show the name of the motion
            font = pygame.font.Font(None, 36)
            text = font.render(motion["name"], True, (0, 0, 0))
            self.screen.blit(text, (20, 20))
            pygame.display.flip()
            pygame.time.wait(int(pause_time * 1000))

    def draw_trajectory(self, trajectory, color=(200, 0, 200)):
        """Draws the trajectory as a line on the screen."""
        scale_x = self.screen.get_width() / self.map_width
        scale_y = self.screen.get_height() / self.map_height
        points = [
            (int(x * scale_x), int(self.screen.get_height() - y * scale_y))
            for x, y in trajectory
        ]
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 3)



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0  # Reset step counter
        
        # Set random orientation
        random_theta = self.np_random.uniform(-math.pi, math.pi)  
        
        # Reset robot
        self.robot.reset(self.start_x, self.start_y, random_theta)
        
        return self.robot.get_state(), {}  # Return initial state and empty info
    
    def step(self, action):
        self.step_count += 1
        left_voltage, right_voltage = action  # Extract actions
        
        # Update robot
        collision = self.robot.update(left_voltage, right_voltage, self.dt, self.walls, self.goal)
        
        # Get current state
        state = self.robot.get_state()
        
        # Check if goal is reached or collision occurred
        done = bool(self.robot.color_sensor) or collision
        truncated = (self.step_count >= self.max_steps)
        
        # Create info dict
        info = {
            'at_goal': bool(self.robot.color_sensor),
            'collision': collision,
            'timeout': truncated
        }
        
        return state, done, truncated, info
    
    def render(self):
        if self.render_mode is None:
            return
            
        # Initialize Pygame if not already initialized
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            
            # Set screen size based on map dimensions
            screen_width = int(self.map_width * 800 / 2)
            screen_height = int(self.map_height * 800 / 2)
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            self.clock = pygame.time.Clock()
            self.isopen = True
            
            # Set caption
            if self.map_type == MapType.STRAIGHT:
                pygame.display.set_caption("Robot Navigation - Straight Corridor")
            else:
                pygame.display.set_caption("Robot Navigation - T-Junction")
        
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Scale factors for drawing
        scale_x = self.screen.get_width() / self.map_width
        scale_y = self.screen.get_height() / self.map_height
        
        # Draw walls
        for wall in self.walls:
            x1, y1, x2, y2 = wall
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                (int(x1 * scale_x), int(self.screen.get_height() - y1 * scale_y)),
                (int(x2 * scale_x), int(self.screen.get_height() - y2 * scale_y)),
                3
            )
        
        # Draw goal as a rectangle
        goal_x, goal_y, goal_width, goal_height = self.goal
        half_w = goal_width / 2
        half_h = goal_height / 2
        pygame.draw.rect(
            self.screen,
            (0, 255, 0),
            (
                int((goal_x - half_w) * scale_x), 
                int(self.screen.get_height() - (goal_y + half_h) * scale_y),
                int(goal_width * scale_x),
                int(goal_height * scale_y)
            )
        )
        
        # Draw robot
        robot_x = int(self.robot.x * scale_x)
        robot_y = int(self.screen.get_height() - self.robot.y * scale_y)
        robot_radius_px = int(self.robot.radius * scale_x)
        
        # Draw robot body
        pygame.draw.circle(
            self.screen,
            (0, 0, 255),  # Blue
            (robot_x, robot_y),
            robot_radius_px
        )
        
        # Draw robot heading
        heading_x = robot_x + robot_radius_px * math.cos(self.robot.theta)
        heading_y = robot_y - robot_radius_px * math.sin(self.robot.theta)
        pygame.draw.line(
            self.screen,
            (255, 0, 0),  # Red
            (robot_x, robot_y),
            (int(heading_x), int(heading_y)),
            2
        )
        
        # Draw sensors starting from the robot's circumference
        sensor_angles = [
            self.robot.theta + self.robot.left_sensor_angle,
            self.robot.theta + self.robot.front_sensor_angle,
            self.robot.theta + self.robot.right_sensor_angle
        ]
        
        sensor_values = [self.robot.left_sensor, self.robot.front_sensor, self.robot.right_sensor]
        
        for i, (angle, value) in enumerate(zip(sensor_angles, sensor_values)):
            # Calculate start point on robot circumference
            start_x = robot_x + int(robot_radius_px * math.cos(angle))
            start_y = robot_y - int(robot_radius_px * math.sin(angle))
            
            # Calculate end point
            end_x = robot_x + int((robot_radius_px + value * scale_x) * math.cos(angle))
            end_y = robot_y - int((robot_radius_px + value * scale_y) * math.sin(angle))
            
            # Different colors for each sensor
            colors = [(255, 0, 255), (255, 165, 0), (0, 255, 255)]  # Magenta, Orange, Cyan
            
            pygame.draw.line(
                self.screen,
                colors[i],
                (start_x, start_y),
                (end_x, end_y),
                2
            )
        
        # Draw status information
        font = pygame.font.Font(None, 24)
        
        # Goal detection
        if self.robot.color_sensor == 1:
            text = font.render("Goal Detected!", True, (0, 200, 0))
            self.screen.blit(text, (10, 10))
        
        # Vehicle info
        info_y = self.screen.get_height() - 100
        
        # Display voltages
        voltage_text = font.render(
            f"Voltages - Left: {self.robot.left_voltage:.2f}V, Right: {self.robot.right_voltage:.2f}V",
            True, (0, 0, 0)
        )
        self.screen.blit(voltage_text, (10, info_y))
        
        # Display velocities
        vel_text = font.render(
            f"Velocities - v: {self.robot.v:.2f} m/s, w: {self.robot.w:.2f} rad/s",
            True, (0, 0, 0)
        )
        self.screen.blit(vel_text, (10, info_y + 25))
        
        # Display sensor readings
        sensor_text = font.render(
            f"Sensors - Left: {self.robot.left_sensor:.2f}m, Front: {self.robot.front_sensor:.2f}m, Right: {self.robot.right_sensor:.2f}m",
            True, (0, 0, 0)
        )
        self.screen.blit(sensor_text, (10, info_y + 50))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(1 / self.dt)  # Limit FPS to simulation speed
        
        return self.screen
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.isopen = False




# Test the environment
if __name__ == "__main__":
    # Create environment
    # env = RobotEnv(map_type=MapType.T_JUNCTION)
    # env = RobotEnv(map_type=MapType.STRAIGHT)
    # env = RobotEnv(map_type=MapType.NARROW)
    env = RobotEnv(map_type=MapType.PHYSICS)
    env.demo_trajectories()
    env.close()
    
    
    env.close()
