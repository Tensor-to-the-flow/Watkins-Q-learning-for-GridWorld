import gym
import numpy as np
import ray
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from really import SampleManager
from gridworlds import GridWorld

"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""


class TabularQ(object):
    def __init__(self, h, w, action_space):
        self.action_space = action_space
        # randomly initialize q-values, 0 for terminal or unreachable states
        # 3D array: y_pos,x_pos,action_index
        self.q_values = np.zeros(shape=(h,w,action_space))

    def __call__(self, state):
        ## # TODO:
        # expand by batch size dim to q values
        state = np.squeeze(state, axis=0)
        output = {}
        # choose action that maximizes q value of s', observe reward and s'
        # return q-values for state
        output["q_values"] = self.q_values[int(state[0]),int(state[1])]
        output["q_values"] = np.expand_dims(output["q_values"],axis=0)
        return output

    def get_weights(self):
        weights = np.copy(self.q_values)
        return weights

    def set_weights(self, q_vals):
        self.q_values = q_vals

    def q_val(self, state, action):
        model_out = self(state)
        q_values = model_out["q_values"]
        q_val = q_values[0,action]
        return q_val

    def max_q(self, state):
        # compute maximum q value along each batch dimension
        model_out = self(state)
        q_values = model_out["q_values"]
        x = np.max(q_values, axis=1)
        return x

    def update(self, sample_dict):
        losses = []
        for step in np.arange(len(sample_dict["not_done"])):
            nd = sample_dict["not_done"][step]
            s = sample_dict["state"][step]
            a = sample_dict["action"][step]
            r = sample_dict["reward"][step]
            s_new = sample_dict["state_new"][step]
            # if nonterminal:
                # get q values for actions in new state
                # Q(s,a)+α[r(s,a)+γmaxa′Q(s′,a′)−Q(s,a)]
            old_q_value = self.q_val(np.expand_dims(s,axis=0),a)
            # calculate new q-values for actions in current state
            q_value = old_q_value + alpha*(r+gamma*self.max_q(np.expand_dims(s_new, axis=0))*nd-old_q_value)

            new_weights = self.get_weights()
            new_weights[int(s[0]),int(s[1]),a] = q_value
            self.set_weights(new_weights)
                # update optimized agent
            losses.append((old_q_value-q_value)**2)
            # if (sample_dict["not_done"][step] == 0):
            #     print("got reward: ",r)
                #break
        return np.mean(np.asarray(losses))

    def save(self, full_path):
        # Reshape to 3D to 2D array
        model = self.q_table.reshape(self.q_table.shape[0], -1)
        # Save as csv
        np.savetxt(full_path, model, delimiter=",")

    def load(self, saving_path):
        # load the csv
        model = np.loadtxt(saving_path, delimiter=",")
        #  Reshape 2D to 3D
        model = model.reshape(
            model.shape[0], model.shape[1] // self.q_table.shape[2], self.q_table.shape[2])
        self.set_weights(model)


if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 10,
        "width": 10,
        "action_dict": action_dict,
        "start_position": (0, 0),
        "reward_position": (5, 8),
    }

    # create environment after installation: env = gym.make('gridworld-v0')
    env = GridWorld(**env_kwargs)

    epsilon = 1
    min_epsilon = 0.01
    decay_factor = 0.95
    alpha = 0.8
    gamma = 0.95
    buffer_size = 5000
    test_steps = 50
    epochs = 30
    sample_size = 1000
    show_every = 30
    delta = 0.000000000001 # size of mean update error

    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 4,
        "total_steps": 1000,
        "model_kwargs": model_kwargs,
        "env_kwargs": env_kwargs,
        # and more: action sampling strategy
        "action_sampling_type": "epsilon_greedy",
        "epsilon": 1,
    }

    # initilize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    # where to save results
    saving_path = os.getcwd() + "\progress_tabularq"

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # this was given
    print("test before training: ")
    manager.test(
        max_steps=50,
        test_episodes=5,
        #render=True,
        #do_print=True,
        evaluation_measure="time_and_reward",
    )

    # get initial agent
    # agent = manager.get_agent()
    losses = []

    for epoch in range(epochs):
        # training core
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)
        # sample data to optimize on
        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        print("optimizing...")
        # get agent to iterate through trajectory data
        agent = manager.get_agent()
        losses.append(agent.model.update(sample_dict))
        new_weights = agent.model.get_weights()
        manager.set_agent(new_weights)
        # Reduce epsilon if updates were performed (because we need less and less exploration)
        if losses[-1] != 0:
            epsilon = epsilon*decay_factor
            manager.set_epsilon(epsilon)

        time_steps = manager.test(test_steps,test_episodes=10, render=(epoch%show_every==0))
        # update aggregator --> saving error
        # manager.update_aggregator(loss=losses[-1], time_steps=time_steps)
        print(f" epoch ::: {epoch}  loss ::: {losses[-1]}   avg env steps ::: {np.mean(time_steps)}")
        # stop if error in updates was smaller than treshold
        # if (losses[-1] > 0) & (losses[-1] < delta):
        #     break
        # #if epoch % saving_after == 0:
            # you can save models
        #    manager.save_model(saving_path, e)

    # and load models
    # manager.load_model(saving_path)
    # print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True, do_print=True)

# watkins q-learning algorithm
# step size alpha in (0,1], small epsilon > 0
# initialize q values q(s,a) for all s,a arbitrarily except terminal s = 0
# for each episode:
#   initialize s
#   for each step in episode:
#       choose a from s using e.g. epsilon-greedy policy according to q-value_estimates
#       take a, observe reward, s'
#       update q(s,a) = q(s,a)+alpha[reward+gamma* max_a' of q(s',a)-q(s,a)]
#       s = s'
#   until s is terminal

# Updates are performed by combining one step of sampling for the q-value
# with a greedy TD estimate:
# δTD−Watkins−Q = r(s,a) + γ * maxa′ Q(s′,a′) − Q(s,a)
