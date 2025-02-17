class PID:
    def __init__(
            self, gain_prop: int, gain_int: int, gain_der: int, sensor_period: float,
            output_limits: tuple[float, float]
            ):
        self.gain_prop = gain_prop
        self.gain_der = gain_der
        self.gain_int = gain_int
        self.sensor_period = sensor_period
        # TODO: define additional attributes you might need
        self.output_limits = output_limits
        self.accumulated_error = 0
        self.previous_error = 0
        # END OF TODO


    # TODO: implement function which computes the output signal
    # The controller should output only in the range of output_limits
    def output_signal(self, commanded_variable: float, sensor_readings: list[float]) -> float:
        current_value = sensor_readings[0]

        error = commanded_variable - current_value
        prop_term = self.gain_prop * error

        self.accumulated_error += error * self.sensor_period
        int_term = self.gain_int * self.accumulated_error

        der = (error - self.previous_error) / self.sensor_period
        der_term = self.gain_der * der

        output_signal = prop_term + int_term + der_term

        self.previous_error = error
        lower_bound, upper_bound = self.output_limits
        
        output_signal = max(output_signal, lower_bound)
        output_signal = min(output_signal, upper_bound)
        #print(prop_term,der_term)
        return output_signal
    # END OF TODO
