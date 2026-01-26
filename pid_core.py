# pid_core.py

class PID:
    def __init__(self, kp, ki, kd, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.prev_error = 0.0
        self.integral_limit = integral_limit

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, setpoint, measurement, dt):
        error = setpoint - measurement

        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = max(
                -self.integral_limit,
                min(self.integral, self.integral_limit)
            )

        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        return (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )
