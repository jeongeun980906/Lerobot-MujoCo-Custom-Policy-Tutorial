import zmq
import numpy as np
import time
import threading

class controller:
    def __init__(self, mode='joint'):
        self.mode = mode
        self.context_ = zmq.Context()
        self.socket = self.context_.socket(zmq.SUB)
        self.socket.bind("tcp://*:5555")  # bind
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        
        self.table_height = 0.3
        self.action_data = None  # Will store the latest (p, R, gripper) tuple
        
        # Flag for running the listener thread
        self.keep_running = True
        
        # Start a thread to continuously read the data
        self.thread = threading.Thread(target=self._data_listener)
        self.thread.daemon = True  # Optional: makes the thread exit when the main program exits
        self.thread.start()
        
    def _data_listener(self):
        """Thread method that continuously reads incoming data."""
        while self.keep_running:
            try:
                # Attempt a non-blocking read from the socket
                data = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
                # print(data)
                # Unpack the received dictionary
                p, R, gripper, q = data['p'], data['R'], data['gripper'], data['qpos']
                # print(q)
                # Process the pose data
                if self.mode == 'eef_pose':
                    p = self.refine_pose(p)
                    # Store the processed data for later retrieval
                    self.action_data = (p, R, gripper)
                elif self.mode == 'joint':
                    self.action_data = q
                else: 
                    raise NotImplementedError
            except zmq.Again:
                # No message received; sleep briefly to reduce CPU load
                time.sleep(0.01)
    
    def get_action(self):
        """Return the most recent action data received.
        
        Returns:
            tuple or None: The latest (p, R, gripper) data, or None if no data has been received.
        """
        return self.action_data
    
    def refine_pose(self, p):
        """Refine the incoming pose by adjusting the z-coordinate.
        
        Args:
            p (list or np.array): Pose represented with at least 3 values (x, y, z).
            
        Returns:
            The refined pose.
        """
        if self.mode == 'right_only':
            p -= np.array([ 0.05751911, -0.0425337, 0.0])
            p[0] *= 1.5
            p[0] += 0.5
            p[1] *= 2.3
            p[1] += 0.3
            p[2] *= 3.5
            p[2] += self.table_height
            return p
        else:
            raise NotImplementedError
    
    def reset(self, env):
        # Implement your reset logic here if needed.
        pass
    
    def stop(self):
        """Stop the data listener thread safely."""
        self.keep_running = False
        self.thread.join()