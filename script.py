import h5py
import numpy as np

f = h5py.File("data/SafetyPointGoal1Gymnasium-v0-100-2022.hdf5", "r")

# 1. What arrays are inside?
print(list(f.keys()))

# 2. How big is each one?
for name in f.keys():
    # print(name, f[name].shape, f[name].dtype)
    pass

# 3. Look at the first 10 rewards and costs
# print(f["rewards"][:10])
# print(f["costs"][:10])


rewards = f["rewards"][0:1000]
costs = f["costs"][0:1000]

# print("total reward:", rewards.sum())
# print("total cost:  ", costs.sum())
# print("steps where a goal was reached:", np.flatnonzero(rewards > 0.9))
# print("first 5 costly steps:", np.flatnonzero(costs > 0)[:5])

obs = f["observations"]
nxt = f["next_observations"]

for k in [1, 4]:
    start = k * 1000
    last_of_previous = nxt[start - 1]   # final state of episode k-1
    first_of_this = obs[start]          # recorded first observation of episode k
    after_first_step = nxt[start]       # what the robot sensed after its first action
    print(f"episode {k}")
    print("  first obs == previous episode's last state?", np.allclose(last_of_previous, first_of_this))
    print("  hazard lidar change over the first step:", np.abs(first_of_this[28:44] - after_first_step[28:44]).sum().round(2))
def find_runs(mask):
    padded = np.concatenate(([0], mask.astype(int), [0]))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return starts, stops


costs = f["costs"][0:1000]
starts, stops = find_runs(costs > 0)
print("stays (start, stop, duration):")
for a, b in zip(starts, stops):
    print(" ", a, b, b - a)
print("total stay steps:", (stops - starts).sum())

closeness = f["next_observations"][0:1000, 28:44].max(axis=1)
print("agreement:", np.mean((closeness >= 0.933) == (costs > 0)))
