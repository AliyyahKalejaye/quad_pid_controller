from controller import Robot
import sys
import math
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import threading
from queue import Queue
import time
import pygame
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import warnings
# Suppress only matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Or suppress specific warning messages
warnings.filterwarnings('ignore', message='Starting a Matplotlib GUI outside of the main thread')

def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)




class ThreadedDataPlotter:
    def __init__(self, max_points=50, update_interval=5):
        self.max_points = max_points
        self.update_interval = update_interval
        self.data_queue = Queue()
        self.running = True
        
        # Start plotting thread
        self.plot_thread = threading.Thread(target=self._plotting_loop)
        self.plot_thread.daemon = True
        self.plot_thread.start()
        
        # Wait for plot initialization
        self.initialized = threading.Event()
        self.data_queue.put(('init', None))
        self.initialized.wait()

    def _plotting_loop(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 6))

    # Add this line to increase responsiveness
        self.fig.canvas.flush_events()
        
        # Increase animation speed
        plt.rcParams['animation.html'] = 'html5'
        
        # Reduce plot overhead
        self.fig.set_dpi(80)  # Lower DPI for faster rendering
        plt.style.use('fast')  # Use fast style

        plt.tight_layout()
        
        # Initialize data containers
        self.times = deque(maxlen=self.max_points)
        self.rolls = deque(maxlen=self.max_points)
        self.pitches = deque(maxlen=self.max_points)
        self.yaws = deque(maxlen=self.max_points)
        self.altitudes = deque(maxlen=self.max_points)
        self.roll_inputs = deque(maxlen=self.max_points)
        self.pitch_inputs = deque(maxlen=self.max_points)
        self.yaw_inputs = deque(maxlen=self.max_points)
        
        # Initialize plot lines
        self.roll_line, = self.ax1.plot([], [], 'r-', label='Roll')
        self.pitch_line, = self.ax1.plot([], [], 'g-', label='Pitch')
        self.yaw_line, = self.ax1.plot([], [], 'b-', label='Yaw')
        self.altitude_line, = self.ax2.plot([], [], 'k-', label='Altitude')
        self.roll_input_line, = self.ax3.plot([], [], 'r--', label='Roll Input')
        self.pitch_input_line, = self.ax3.plot([], [], 'g--', label='Pitch Input')
        self.yaw_input_line, = self.ax3.plot([], [], 'b--', label='Yaw Input')
        
        # Setup axes
        self.ax1.set_title('Attitude')
        self.ax1.set_ylabel('Angles (deg)')
        self.ax1.legend()
        self.ax1.grid(True)
        
        self.ax2.set_title('Altitude')
        self.ax2.set_ylabel('Height (m)')
        self.ax2.legend()
        self.ax2.grid(True)
        
        self.ax3.set_title('Control Inputs')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Input')
        self.ax3.legend()
        self.ax3.grid(True)
        
        self.initialized.set()
        
        last_update = time.time()
        while self.running:
            try:
                cmd, data = self.data_queue.get(timeout=0.1)
                if cmd == 'update':
                    t, roll, pitch, yaw, altitude, roll_input, pitch_input, yaw_input = data
                    self._update_data(t, roll, pitch, yaw, altitude, roll_input, pitch_input, yaw_input)
                    
                    current_time = time.time()
                    if current_time - last_update >= self.update_interval / 1000.0:
                        self._update_plot()
                        last_update = current_time
                elif cmd == 'stop':
                    break
            except Queue.Empty:
                continue

    def _update_data(self, t, roll, pitch, yaw, altitude, roll_input, pitch_input, yaw_input):
        self.times.append(t)
        self.rolls.append(roll)
        self.pitches.append(pitch)
        self.yaws.append(yaw)
        self.altitudes.append(altitude)
        self.roll_inputs.append(roll_input)
        self.pitch_inputs.append(pitch_input)
        self.yaw_inputs.append(yaw_input)

    def _update_plot(self):
        self.roll_line.set_data(list(self.times), list(self.rolls))
        self.pitch_line.set_data(list(self.times), list(self.pitches))
        self.yaw_line.set_data(list(self.times), list(self.yaws))
        self.altitude_line.set_data(list(self.times), list(self.altitudes))
        self.roll_input_line.set_data(list(self.times), list(self.roll_inputs))
        self.pitch_input_line.set_data(list(self.times), list(self.pitch_inputs))
        self.yaw_input_line.set_data(list(self.times), list(self.yaw_inputs))
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.relim()
            ax.autoscale_view()
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update(self, t, roll, pitch, yaw, altitude, roll_input, pitch_input, yaw_input):
        self.data_queue.put(('update', (t, roll, pitch, yaw, altitude, roll_input, pitch_input, yaw_input)))

    def close(self):
        self.running = False
        self.data_queue.put(('stop', None))
        self.plot_thread.join()
        plt.close(self.fig)






class PIDController:
    def __init__(self, kp, ki, kd, min_output=-10, max_output=10):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.integral = 0
        self.previous_error = 0
        self.first_update = True

    def compute(self, error, dt):
        if dt <= 0:
            return 0

        p_term = self.kp * error #p term
        
        self.integral = clamp(self.integral + error * dt, -5, 5) #iterm
        i_term = self.ki * self.integral 

        if self.first_update:
            d_term = 0
            self.first_update = False
        else:
            d_term = self.kd * (error - self.previous_error) / dt #d_term
        
        self.previous_error = error
        output = clamp(p_term + i_term + d_term, self.min_output, self.max_output)
        return output
    


class Mavic(Robot):
    K_VERTICAL_THRUST = 70.0
    K_VERTICAL_OFFSET = 0.6

    def __init__(self):
        Robot.__init__(self)
        self.time_step = int(self.getBasicTimeStep())
        
        # Initialize pygame and joystick
        pygame.init()
        pygame.joystick.init()

          # Check if any joystick/game controller is connected
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Joystick detected: {self.joystick.get_name()}")
        else:
            print("No joystick detected!")
            self.joystick = None
              # Control parameters
        self.max_tilt = 10.0  # Maximum tilt angle in degrees
        self.max_vertical_speed = 1.1  # Maximum vertical speed in m/s
        self.max_yaw_rate = 90.0  # Maximum yaw rate in degrees/s

        
        

        # Initialize sensors
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        # Initialize motors
        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")

        motors = [self.front_left_motor, self.front_right_motor, 
                 self.rear_left_motor, self.rear_right_motor]
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0)


        # no control controllers
        
        # self.roll_pid =     PIDController(kp=0, ki=3, kd=10)
        # self.pitch_pid =    PIDController(kp=0.21, ki=7, kd=0.09)     
        # self.yaw_pid =      PIDController(kp=6, ki=0.15, kd=1)
        # self.altitude_pid = PIDController(kp=10.0, ki=0.01, kd=5)
      

    

        # Initialize PID controllers original

        # self.roll_pid =     PIDController(kp=0.21, ki=0.1, kd=0.1)
        # self.pitch_pid =    PIDController(kp=0.21, ki=0.2, kd=0.09)     
        # self.yaw_pid =      PIDController(kp=0.13, ki=0.15, kd=0.027)
        # self.altitude_pid = PIDController(kp=10.0, ki=0.01, kd=5)
      

        #optimized gains

        self.roll_pid =     PIDController(kp=0.215, ki=0.197, kd=0.095)
        #self.roll_pid =     PIDController(kp=2.9, ki=1.5, kd=0.0976)
        self.pitch_pid =    PIDController(kp=0.21, ki=0.2, kd=0.09)     
        self.yaw_pid =      PIDController(kp=0.1298, ki=0.015, kd=0.09)
        self.altitude_pid = PIDController(kp=10.0, ki=0.01, kd=5)

     


        self.target_altitude = 1.0  # Initial hover height
        self.altitude_hold = True   # New flag for altitude 
        self.last_time = self.getTime()
        self.altitude_cmd = 0
        self.target_yaw = 0
        
        # Initialize plotter
        self.plotter = ThreadedDataPlotter(max_points=200, update_interval=20)

    
    def get_joystick_input(self):
        """Get normalized joystick inputs between -1 and 1"""
        if not self.joystick:
            return 0.0, 0.0, 0.0, 0.0

        pygame.event.pump()  # Process pygame event queue

         # Left stick (roll and pitch)
        roll_cmd = self.joystick.get_axis(0)  # Left/Right on left stick
        pitch_cmd = -self.joystick.get_axis(1)  # Up/Down on left stick
        
        # Right stick (yaw and altitude)
        yaw_cmd = self.joystick.get_axis(2)  # Left/Right on right stick

    # Apply deadband to yaw command
        if abs(yaw_cmd) < 0.15:  # 15% deadband
            yaw_cmd = 0.0

        # Update target yaw with limits
        if yaw_cmd < -0.5:
            self.target_yaw = (self.target_yaw - 0.5) % 360
        if yaw_cmd > 0.5:
            self.target_yaw = (self.target_yaw + 0.5) % 360
            

        altitude_up = self.joystick.get_axis(5)  
        altitude_down = self.joystick.get_axis(4)
       

        if(altitude_up > 0.5):
            self.altitude_cmd += 0.005       
        if(altitude_down > 0.5):
            self.altitude_cmd -= 0.005

        if self.altitude_cmd <= 0:
                self.altitude_cmd = 0
        if self.altitude_cmd >= 10:
                self.altitude_cmd = 10
                
       
                
        
        # Apply deadzone
        # Apply deadzone with smaller deadband
        roll_cmd = self.apply_deadband(roll_cmd, deadband=0.8)
        pitch_cmd = self.apply_deadband(pitch_cmd, deadband=0.5)
        yaw_cmd = self.apply_deadband(yaw_cmd, deadband=0.8)
        altitude_cmd = self.apply_deadband(self.altitude_cmd, deadband=0.5)
        
        
        return roll_cmd, pitch_cmd, yaw_cmd, self.altitude_cmd

    def apply_deadband(self, value, deadband=0.5):
        """Apply deadband to a value. Returns 0 if value is within ±deadband."""
        return 0 if abs(value) < deadband else value

    def run(self):
        try:
            while self.step(self.time_step) != -1:
                
                current_time = self.getTime()
                dt = current_time - self.last_time if self.last_time != 0 else 0.01
                self.last_time = current_time



                  # Get joystick commands
                roll_cmd, pitch_cmd, yaw_cmd, altitude_cmd = self.get_joystick_input()


                  # Update target values based on joystick input
                target_roll = roll_cmd * self.max_tilt
                target_pitch = pitch_cmd * self.max_tilt
                target_yaw_rate = self.target_yaw



                   # Only update target altitude when stick is moved
                if abs(altitude_cmd) > 0:
                    self.altitude_hold = False
                    self.target_altitude = altitude_cmd
                  
                    self.target_altitude = max(0.1, self.target_altitude)  # Keep minimum altitude
                else:
                    self.altitude_hold = True



                
             

                print(f"xtr => Roll: {target_roll:.2f}, Pitch: {target_pitch:.2f}, Yaw: {target_yaw_rate:.2f}, Altitude: {self.target_altitude:.2f}")
               

                # Get sensor readings
                roll, pitch, yaw = self.imu.getRollPitchYaw()
                _, _, altitude = self.gps.getValues()
                roll_rate, pitch_rate, yaw_rate = self.gyro.getValues()

                # Convert to degrees
                roll = math.degrees(roll)
                pitch = math.degrees(pitch)
                yaw = math.degrees(yaw)
                # yaw = ((yaw + 180) % 360) - 180
                yaw = yaw % 360

                print(f"rpy => Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Altitude: {altitude:.2f}")

                

                  # Calculate errors
                roll_error = roll - target_roll
                pitch_error = pitch - target_pitch
                # yaw_error = target_yaw_rate - yaw  # Use rate control for yaw
                yaw_error = ((self.target_yaw - yaw + 180) % 360) - 180
                
                altitude_error = self.target_altitude - altitude

                
                print(f"Errors => Roll: {roll_error:.2f}, Pitch: {pitch_error:.2f}, Yaw: {yaw_error:.2f}, Altitude: {altitude_error:.2f}")


                # pitchGains = self.adaptiveControl.compute_control_signal(pitch_error)

                # print(pitchGains, "---- >")




                # Compute PID outputs
                roll_input = self.roll_pid.compute(roll_error, dt)
                pitch_input = self.pitch_pid.compute(pitch_error, dt)
                yaw_input = self.yaw_pid.compute(yaw_error, dt)
                altitude_input = self.altitude_pid.compute(altitude_error, dt)
                
                

                # Update plot data
                        # In the run() method of Mavic class
                self.plotter.update(
                    current_time,  # Make sure this is increasing
                    roll, pitch, yaw, altitude,
                    roll_input, pitch_input, yaw_input
                )

                # Calculate motor inputs
                front_left  = self.K_VERTICAL_THRUST +  altitude_input - yaw_input + pitch_input - roll_input  #1
                front_right = self.K_VERTICAL_THRUST +  altitude_input + yaw_input + pitch_input + roll_input  #2
                rear_left   = self.K_VERTICAL_THRUST +  altitude_input + yaw_input - pitch_input - roll_input  #4
                rear_right =  self.K_VERTICAL_THRUST +  altitude_input - yaw_input - pitch_input + roll_input  #3

                # Set motor velocities
                self.front_left_motor.setVelocity(front_left)
                self.front_right_motor.setVelocity(-front_right)
                self.rear_left_motor.setVelocity(-rear_left)
                self.rear_right_motor.setVelocity(rear_right)

                print(f"\nPID Outputs:")
                print(f"Roll PID: {roll_input:.2f}")
                print(f"Pitch PID: {pitch_input:.2f}")
                print(f"Yaw PID: {yaw_input:.2f}")
                print(f"Altitude PID: {altitude_input:.2f}")
                print("-" * 50)
           

        finally:
            self.plotter.close()

# Main execution
if __name__ == "__main__":
    robot = Mavic()
    robot.run()