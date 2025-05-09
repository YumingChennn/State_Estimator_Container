from pathlib import Path
import pinocchio

# Load URDF
model = pinocchio.buildModelFromUrdf(
    "/home/ray/State_Estimator_Container/urdf/reddog.urdf",
    pinocchio.JointModelFreeFlyer()
)

data = model.createData()
print("model name:", model.name)

# Random configuration (or use real sensor data here)
q = pinocchio.randomConfiguration(model)
print("q:", q)

# Run FK
pinocchio.forwardKinematics(model, data, q)
pinocchio.updateFramePlacements(model, data)

# Define the foot frames (modify these names to match your URDF)
foot_frame_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# Print foot positions
for name in foot_frame_names:
    frame_id = model.getFrameId(name)
    pos = data.oMf[frame_id].translation
    print(f"{name} position: {pos}")

print("model.nq =", model.nq)
print("model.nv =", model.nv)

