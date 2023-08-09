import numpy as np

import sys, argparse, time, os

import torch
from cassie import *
from cassie.cassiemujoco import pd_in_t


parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="/home/xyz/projects/cassie_mujoco_rppo/LOG_DIRECTORY/curmodel/actor.pt", type=str)
policy_model = parser.parse_args()
# load the ppo actor model
policy_model = torch.load(policy_model.policy)
policy_model.eval()

# create environment
env = CassieEnv()
env.render()
env.reset()
    
u = pd_in_t()
observations_tensor = torch.Tensor(env.reset())

while True:
    # print(env.phase)
    observations_tensor = torch.Tensor(env.get_full_state())
    if env.phase >= 28:
        env.phase = 0
        env.counter += 1
        #break

    with torch.no_grad():
        actions = policy_model(observations_tensor)

    action_val = actions.numpy().squeeze()

    env.step_simulation(action_val)
    env.sim.step_pd(env.u)
    env.render()



# import numpy as np
# import torch
# from cassie import *
# rom cassie.cassiemujoco import pd_in_t

# import mujoco
# import mujoco_viewer


# parser.add_argument("--policy", default="/home/xyz/projects/cassie_mujoco_rppo/LOG_DIRECTORY/curmodel/actor.pt", type=str)
# policy_model = parser.parse_args()


# model = mujoco.MjModel.from_xml_path('/home/xyz/projects/cassie_mujoco_rppo/cassie/cassiemujoco/cassie.xml')
# data = mujoco.MjData(model)

# # create the viewer object
# viewer = mujoco_viewer.MujocoViewer(model, data)


#  # create environment
# env = CassieEnv()
# env.render()
# env.reset()

# policy_model = torch.load(policy_model.policy)
# policy_model.eval()

# while True:
#     # print(env.phase)
#     observations_tensor = torch.Tensor(env.get_full_state())
#     if env.phase >= 28:
#         env.phase = 0
#         env.counter += 1
#         #break

#     with torch.no_grad():
#         actions = policy_model(observations_tensor)

#     action_val = actions.numpy().squeeze()

#     env.step_simulation(action_val)
#     env.sim.step_pd(env.u)
#     env.render()



# # simulate and render
# for _ in range(10000):
#     if viewer.is_alive:
#         mujoco.mj_step(model, data)
#         viewer.render()
#     else:
#         break

# # close
# viewer.close()







#     # load the ppo actor model


    
#     u = pd_in_t()
#     observations_tensor = torch.Tensor(env.reset())

