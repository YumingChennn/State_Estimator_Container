from ..base import GenericEstimator
import numpy as np

class LinearKFPositionVelocityEstimator(GenericEstimator):
    def __init__(self):
        super().__init__()
        self._xhat = np.zeros((18, 1))  # State vector
        self._A = np.zeros((18, 18))    # System matrix
        self._B = np.zeros((18, 3))     # Control matrix
        self._C = np.zeros((28, 18))    # Observation matrix
        self._P = np.identity(18) * 100 # Covariance matrix
        self._Q0 = np.identity(18)      # Process noise covariance
        self._R0 = np.identity(28)      # Measurement noise covariance
        

    def setup(self):
        self.high_suspect = 100
        dt = self._data.parameters["controller_dt"]

        I3 = np.identity(3)
        Z3 = np.zeros((3, 3))

        # A matrix (18*18)
        self._A.fill(0.0)
        self._A[:3, :3] = I3
        self._A[:3, 3:6] = dt * I3
        self._A[3:6, 3:6] = I3
        self._A[6:, 6:] = np.identity(12)

        # B matrix (18*3)
        self._B[3:6, :] = dt * I3

        # C matrix (28*18)
        C1 = np.hstack([I3, Z3])
        C2 = np.hstack([Z3, I3])

        # Set _C matrix
        self._C[:3, :6] = C1
        self._C[3:6, :6] = C1
        self._C[6:9, :6] = C1
        self._C[9:12, :6] = C1
        self._C[:12, 6:18] = -np.eye(12)  # -1 times Identity matrix
        self._C[12:15, :6] = C2
        self._C[15:18, :6] = C2
        self._C[18:21, :6] = C2
        self._C[21:24, :6] = C2
        self._C[24, 8] = 1
        self._C[25, 11] = 1
        self._C[26, 14] = 1
        self._C[27, 17] = 1

        # Q0 matrix (18*18)
        self._Q0[0:3, 0:3] = (dt / 20.0) * I3
        self._Q0[3:6, 3:6] = (dt * 9.8 / 20.0) * I3
        self._Q0[6:, 6:] = dt * np.eye(12)

        # Initialize Q and R matrices
        self._Q = np.eye(18)
        self._R = np.eye(28)

    def LinearKalmanFilter(self,_xhat, y, _P, a, Q, R):

        _xhat = self._A @ _xhat + self._B @ a 

        At = self._A.T 
        Pm = self._A @ _P @ At + Q  
        Ct = self._C.T 
        yModel = self._C @ _xhat  
        ey = y - yModel 
        S = self._C @ Pm @ Ct + R  

        S_ey = np.linalg.solve(S, ey) # LU

        adjust = Pm @ Ct @ S_ey
        # _xhat_before_adjust = _xhat

        _xhat += adjust  # Update
        # print((Pm @ Ct @ S_ey)[0])

        S_C = np.linalg.solve(S, self._C) 

        # Update the covariance matrix _P
        identity_matrix = np.eye(_P.shape[0])
        _P = (identity_matrix - Pm @ Ct @ S_C) @ Pm

        # Symmetrize _P to avoid numerical instability
        Pt = _P.T
        _P = (_P + Pt) / 2.0

        if np.linalg.det(_P[:2, :2]) > 1e-6:  # Equivalent to C++ `T(0.000001)`
            # Zero out blocks to maintain stability
            _P[:2, 2:] = 0
            _P[2:, :2] = 0

            # Scale the top-left 2x2 block
            _P[:2, :2] /= 10.0

        return _xhat , _P, yModel, ey, adjust

    def run(self):
        params = self._data.parameters
        result = self._data.result

        # get process noise parameter
        process_noise_pimu = params["imu_process_noise_position"]
        process_noise_vimu = params["imu_process_noise_velocity"]
        process_noise_pfoot = params["foot_process_noise_position"]
        sensor_noise_pimu_rel_foot = params["foot_sensor_noise_position"]
        sensor_noise_vimu_rel_foot = params["foot_sensor_noise_velocity"]
        sensor_noise_zfoot = params["foot_height_sensor_noise"]

        # Q matrix
        self._Q[0:3, 0:3] = self._Q0[0:3, 0:3] * process_noise_pimu
        self._Q[3:6, 3:6] = self._Q0[3:6, 3:6] * process_noise_vimu
        self._Q[6:18, 6:18] = self._Q0[6:18, 6:18] * process_noise_pfoot

        # R matrix
        self._R[0:12, 0:12] = self._R0[0:12, 0:12] * sensor_noise_pimu_rel_foot
        self._R[12:24, 12:24] = self._R0[12:24, 12:24] * sensor_noise_vimu_rel_foot
        self._R[24:28, 24:28] = self._R0[24:28, 24:28] * sensor_noise_zfoot
        
        #start
        g = np.array([0, 0, -9.81])
        Rbod = result.rBody.T
        a = result.aWorld + g
        a = a.reshape(-1,1)
        # print(f"a {a}")

        p0 = self._xhat[0:3].reshape(-1)
        v0 = self._xhat[3:6].reshape(-1)

        pzs = np.zeros((4, 1))
        _ps = np.zeros((12, 1))
        _vs = np.zeros((12, 1))

        y = np.zeros((28,1))
        body_pos = np.array(self._data.cheaterState["position"])  # world frame
        body_vel = np.array(self._data.cheaterState["vBody"])     # world frmae

        for i in range(4):
            
            leg = self._data.legControllerData[i]
            p_rel = np.array(leg["p"]).reshape(-1)  # world frame
            dp_rel = np.array(leg["v"]).reshape(-1) # world frame
            omegaBody = np.array(result.omegaBody).reshape(-1) # angular velocity

            # Local position and velocity
            # p_f = Rbod @ p_rel
            p_f = Rbod @ (p_rel - body_pos)
            # temp = np.cross(omegaBody, p_rel) + dp_rel
            # dp_f = Rbod @ temp
            dp_f = Rbod @ (dp_rel - body_vel - np.cross(omegaBody, p_rel - body_pos))

            # print related information
            print(f"[DEBUG Leg {i}] body pos : {body_pos}")
            print(f"[DEBUG Leg {i}] p_rel (foot pos in body frame): {p_rel}")
            print(f"[DEBUG Leg {i}] p_f (foot pos in body frame): {p_f}")
            # print(f"[DEBUG Leg {i}] dp_f (foot vel in body frame): {dp_f}")
            # print("result.contact_estimate[i]",result.contact_estimate[i])
            
            # trust
            phase = np.clip(result.contact_estimate[i], 0.0, 1.0)
            if phase > 0.5:  # in contact
                trust = 1.0
            else:  
                trust = 0.0
            # print("trust",trust)

            idx = i * 3
            qindex = 6 + idx
            rindex1 = idx
            rindex2 = 12 + idx
            rindex3 = 24 + i

            # Update Q block
            self._Q[qindex:qindex + 3, qindex:qindex + 3] *= (1 + (1 - trust) * self.high_suspect)

            # Update R block
            self._R[rindex1:rindex1 + 3, rindex1:rindex1 + 3] *= 1
            self._R[rindex2:rindex2 + 3, rindex2:rindex2 + 3] *= (1.0 + (1.0 - trust) * self.high_suspect)
            self._R[rindex3, rindex3] *= (1.0 + (1.0 - trust) * self.high_suspect)
            
            # Observation y
            _ps[idx:idx+3, 0] = -p_f.reshape(-1)
            _vs[idx:idx+3, 0] = ((1.0 - trust) * v0 + trust * (-dp_f.reshape(-1)))
            pzs[i, 0] = (1.0 - trust) * (p0[2] + p_f[2])
        
        y = np.vstack([
            _ps,
            _vs,
            pzs
        ])

        self._xhat , self._P, yModel, ey, adjust = self.LinearKalmanFilter(self._xhat, y, self._P, a, self._Q, self._R)

        result.position = self._xhat[0:3]
        result.vWorld = self._xhat[3:6]
        result.vBody = result.rBody @ result.vWorld
