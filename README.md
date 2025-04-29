# 🛰️ State Estimator Container

This project implements a **State Estimator Container** in Python, inspired by the original C++ design from [Cheetah Software](https://github.com/mit-biomimetics/Cheetah-Software).

The container integrates three main estimators:
- **Contact Estimator**: Determines whether each foot is in contact with the ground.
- **Orientation Estimator**: Estimates the robot's body orientation using IMU data.
- **Position Velocity Estimator**: Estimates the robot's body position and velocity based on IMU, foot contact, and foot kinematics.

The system takes in:
- IMU measurements
- Foot contact information
- Foot positions and velocities

and outputs:
- Estimated body position
- Estimated body velocity

---

## 📚 References

- **Cheetah Software** (MIT Biomimetic Robotics Lab):  
  https://github.com/mit-biomimetics/Cheetah-Software

- **MiniCheetah State Estimator (Offline Kalman Filter)**:  
  https://github.com/SobhanGhayedzadeh/MiniCheetah_State_Estimator

---

## 🚀 How to Run

```bash
python test_orientation_estimator.py
python test_posvel_estimator.py
