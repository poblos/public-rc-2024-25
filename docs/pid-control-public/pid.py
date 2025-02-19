# TODO: implement a class for PID controller
class PID:
    def __init__(self, gain_prop, gain_int, gain_der, sensor_period):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period

        # TODO: add aditional variables to store the current state of the controller
        self.accumulated_error = 0
        self.previous_error = 0

    # TODO: implement function which computes the output signal
    def output_signal(self, commanded_variable, sensor_readings):
        
        current_value = sensor_readings[0]

        error = commanded_variable - current_value
        prop_term = self.gain_prop * error

        self.accumulated_error += error * self.sensor_period
        int_term = self.gain_int * self.accumulated_error

        der = (error - self.previous_error) / self.sensor_period
        der_term = self.gain_der * der

        output_signal = prop_term + int_term + der_term

        self.previous_error = error

        return output_signal