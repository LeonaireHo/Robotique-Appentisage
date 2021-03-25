import numpy as np
from toolbox import softmax, egreedy_loc, egreedy
from maze import build_maze
import matplotlib.pyplot as plt

from dynamic_programming import policy_iteration_q, get_policy_from_q, create_maze


# -------------------------------------------------------------------------------------#
# Given state and action spaces and a policy, computes the state value of this policy


def temporal_difference(mdp, pol, nb_episodes=50, alpha=0.2, timeout=25, render=True):
    # alpha: learning rate
    # timeout: timeout of an episode (maximum number of timesteps)
    v = np.zeros(mdp.nb_states)  # initial state value v
    mdp.timeout = timeout

    if render:
        mdp.new_render()
    
    for _ in range(nb_episodes):  # for each episode
        
        # Draw an initial state randomly (if uniform is set to False, the state is drawn according to the P0 
        #                                 distribution)
        x = mdp.reset(uniform=True) 
        done = mdp.done()
        while not done:  # update episode at each timestep
            # Show agent
            if render:
                mdp.render(v, pol)
            
            # Step forward following the MDP: x=current state, 
            #                                 pol[i]=agent's action according to policy pol, 
            #                                 r=reward gained after taking action pol[i], 
            #                                 done=tells whether the episode ended, 
            #                                 and info gives some info about the process
            [y, r, done, _] = mdp.step(egreedy_loc(pol[x], mdp.action_space.size, epsilon=0.2))
            # Update the state value of x
            if x in mdp.terminal_states:
                v[x] = r
            else:
                delta = r + mdp.gamma * v[y] - v[x]
                v[x] = v[x] + alpha * delta

            # Update agent's position (state)
            x = y
    
    if render:
        # Show the final policy
        mdp.current_state = 0
        mdp.render(v, pol)
    return v

# --------------------------- Q-Learning -------------------------------#

# Given a temperature "tau", the QLearning function computes the state action-value function
# based on a softmax policy
# alpha is the learning rate


def q_learning_soft(mdp, tau,gamma, nb_episodes=20, timeout=50, alpha=0.5, render=True):
    # Initialize the state-action value function
    # alpha is the learning rate
    q = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_min = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_list = []

    # Run learning cycle
    mdp.timeout = timeout  # episode length

    if render:
        mdp.new_render()

    for _ in range(nb_episodes):
        # print(i)
        # Draw the first state of episode i using a uniform distribution over all the states
        x = mdp.reset(uniform=True)
        done = mdp.done()
        while not done:
            if render:
                # Show the agent in the maze
                mdp.render(q, q.argmax(axis=1))

            # Draw an action using a soft-max policy
            u = mdp.action_space.sample(prob_list=softmax(q, x, tau))

            # Perform a step of the MDP
            [y, r, done, _] = mdp.step(u)

            # Update the state-action value function with q-Learning
            # TODO
            if x in mdp.terminal_states:
                q[x, u] = r
            else:
                delta = r + gamma * np.max(q[y]) - q[x,u]
                q[x, u] = q[x,u] + alpha*delta

            # Update the agent position
            x = y
        q_list.append(np.linalg.norm(np.maximum(q, q_min)))

    if render:
        # Show the final policy
        mdp.current_state = 0
        mdp.render(q, get_policy_from_q(q))
    return q, q_list

def q_learning_eps(mdp, epsilon, nb_episodes=20, timeout=50, alpha=0.5, render=True):
    # Initialize the state-action value function
    # alpha is the learning rate
    q = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_min = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_list = []
    # Run learning cycle
    mdp.timeout = timeout  # episode length
    if render:
        mdp.new_render()
    for _ in range(nb_episodes):
        # Draw the first state of episode i using a uniform distribution over all the states
        x = mdp.reset(uniform=True)
        done = mdp.done()
        while not done:
            if render:
                # Show the agent in the maze
                mdp.render(q, q.argmax(axis=1))
            # Draw an action using a egreedy policy
            u = egreedy(q, x, epsilon)

            # Perform a step of the MDP
            [y, r, done, _] = mdp.step(u)

            # Update the state-action value function with q-Learning
            if x in mdp.terminal_states:
                q[x, u] = r
            else:
                delta = r + mdp.gamma * np.max(q[y]) - q[x,u]
                q[x, u] = q[x,u] + alpha*delta
            # Update the agent position
            x = y
        q_list.append(np.linalg.norm(np.maximum(q, q_min)))

    if render:
        # Show the final policy
        mdp.current_state = 0
        mdp.render(q, get_policy_from_q(q))
    return q, q_list
#sarsa with softmax
def sarsa_soft(mdp, tau,gamma, nb_episodes=20, timeout=50, alpha=0.5, render=True):
    # Initialize the state-action value function
    # alpha is the learning rate
    q = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_min = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_list = []

    # Run learning cycle
    mdp.timeout = timeout  # episode length

    if render:
        mdp.new_render()

    for i in range(nb_episodes):
        # print(i)
        # Draw the first state of episode i using a uniform distribution over all the states
        x = mdp.reset(uniform=True)
        ux = 0
        done = mdp.done()
        while not done:
            if render:
                # Show the agent in the maze
                mdp.render(q, q.argmax(axis=1))
            # Draw an action using a soft-max policy
            u = mdp.action_space.sample(prob_list=softmax(q, x, tau))
            # Perform a step of the MDP
            [y, r, done, _] = mdp.step(u)
            # Update the state-action value function with q-Learning
            if x in mdp.terminal_states:
                q[x, u] = r
            else:
                uy = mdp.action_space.sample(prob_list=softmax(q, y, tau))
                delta = r + gamma * q[y,uy] - q[x,u]
                q[x, u] = q[x,u] + alpha*delta
            # Update the agent position
            x = y
        q_list.append(np.linalg.norm(np.maximum(q, q_min)))
    if render:
        # Show the final policy
        mdp.current_state = 0
        mdp.render(q, get_policy_from_q(q))
    return q, q_list

#sarsa with egreedy
def sarsa_eps(mdp, epsilon, nb_episodes=20, timeout=50, alpha=0.5, render=True):
    # Initialize the state-action value function
    # alpha is the learning rate
    q = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_min = np.zeros((mdp.nb_states, mdp.action_space.size))
    q_list = []
    # Run learning cycle
    mdp.timeout = timeout  # episode length
    if render:
        mdp.new_render()
    for i in range(nb_episodes):
        print(i)
        # Draw the first state of episode i using a uniform distribution over all the states
        x = mdp.reset(uniform=True)
        ux = 0
        done = mdp.done()
        while not done:
            if render:
                # Show the agent in the maze
                mdp.render(q, q.argmax(axis=1))
            # Draw an action using a egreedy policy
            u = egreedy(q, x, epsilon)
            # Perform a step of the MDP
            [y, r, done, _] = mdp.step(u)
            # Update the state-action value function with q-Learning
            if x in mdp.terminal_states:
                q[x, u] = r
            else:
                uy = egreedy(q, y, epsilon)
                delta = r + mdp.gamma * q[y,uy] - q[x,u]
                q[x, u] = q[x,u] + alpha*delta
            # Update the agent position
            x = y
        q_list.append(np.linalg.norm(np.maximum(q, q_min)))
    if render:
        # Show the final policy
        mdp.current_state = 0
        mdp.render(q, get_policy_from_q(q))
    return q, q_list

# -------- plot learning curves of Q-Learning and Sarsa with α,τ and γ----------#
def plot_ql_sarsa_para(m, epsilon, tau, nb_episodes, timeout, alpha, render):
  # alpha = 0.1
    tau = 0.1
    gamma = 0.2
    q, q_list2 = q_learning_soft(m, tau,gamma, nb_episodes, timeout, alpha, render)
    q, q_list4 = sarsa_soft(m, tau,gamma, nb_episodes, timeout, alpha, render)
    # alpha = 0.2
    # tau = 0.8
    gamma = 0.3
    q, q_list22 = q_learning_soft(m, tau,gamma, nb_episodes, timeout, alpha, render)
    q, q_list42 = sarsa_soft(m, tau,gamma, nb_episodes, timeout, alpha, render)
    # alpha = 0.3
    # tau = 1
    gamma = 0.5
    q, q_list23 = q_learning_soft(m, tau,gamma, nb_episodes, timeout, alpha, render)
    q, q_list43 = sarsa_soft(m, tau,gamma, nb_episodes, timeout, alpha, render)
    # alpha = 0.5
    # tau = 5
    gamma = 0.7
    q, q_list24 = q_learning_soft(m, tau,gamma, nb_episodes, timeout, alpha, render)
    q, q_list44 = sarsa_soft(m, tau,gamma, nb_episodes, timeout, alpha, render)
    gamma = 0.9
    # alpha = 1
    # tau = 10
    q, q_list25 = q_learning_soft(m, tau,gamma, nb_episodes, timeout, alpha, render)
    q, q_list45 = sarsa_soft(m, tau,gamma, nb_episodes, timeout, alpha, render)
    #camparison

    plt.plot(range(len(q_list2)), q_list2, label='q-learning gamma = 0.1')
    plt.plot(range(len(q_list22)), q_list22, label='q-learning gamma = 0.3')
    plt.plot(range(len(q_list23)), q_list23, label='q-learning gamma = 0.5')
    plt.plot(range(len(q_list24)), q_list24, label='q-learning gamma = 0.7')
    plt.plot(range(len(q_list25)), q_list25, label='q-learning gamma = 0.9')
    plt.xlabel('Number of episodes')
    plt.ylabel('Norm of Q values')
    plt.legend(loc='upper right')
    plt.savefig("comparison_Param_Gamma_Q.png")
    plt.clf()
    plt.plot(range(len(q_list4[200:])), q_list4[200:], label='sarsa gamma = 0.1',linestyle = "--")
    plt.plot(range(len(q_list42[200:])), q_list42[200:], label='sarsa gamma = 0.3',linestyle = "--")
    plt.plot(range(len(q_list43[200:])), q_list43[200:], label='sarsa gamma = 0.5',linestyle = "--")
    plt.plot(range(len(q_list44[200:])), q_list44[200:], label='sarsa gamma = 0.7',linestyle = "--")
    plt.plot(range(len(q_list45[200:])), q_list45[200:], label='sarsa gamma = 0.9',linestyle = "--")
    plt.xlabel('Number of episodes')
    plt.ylabel('Norm of Q values')
    plt.legend(loc='upper right')
    plt.savefig("comparison_Param_Gamma_S.png")
    plt.clf()

# -------- plot learning curves of Q-Learning and Sarsa using epsilon-greedy and softmax ----------#
def plot_ql_sarsa(m, epsilon, tau, nb_episodes, timeout, alpha, render):
    epsilon = 0.001
    tau = 0.1
    q, q_list1 = q_learning_eps(m, epsilon, nb_episodes, timeout, alpha, render)
    q, q_list2 = q_learning_soft(m, tau, nb_episodes, timeout, alpha, render)
    q, q_list3 = sarsa_eps(m, epsilon, nb_episodes, timeout, alpha, render)
    q, q_list4 = sarsa_soft(m, tau, nb_episodes, timeout, alpha, render)
    epsilon = 0.01
    tau = 5
    q, q_list12 = q_learning_eps(m, epsilon, nb_episodes, timeout, alpha, render)
    q, q_list22 = q_learning_soft(m, tau, nb_episodes, timeout, alpha, render)
    q, q_list32 = sarsa_eps(m, epsilon, nb_episodes, timeout, alpha, render)
    q, q_list42 = sarsa_soft(m, tau, nb_episodes, timeout, alpha, render)
    epsilon = 0.1
    tau = 10
    q, q_list13 = q_learning_eps(m, epsilon, nb_episodes, timeout, alpha, render)
    q, q_list23 = q_learning_soft(m, tau, nb_episodes, timeout, alpha, render)
    q, q_list33 = sarsa_eps(m, epsilon, nb_episodes, timeout, alpha, render)
    q, q_list43 = sarsa_soft(m, tau, nb_episodes, timeout, alpha, render)
    #camparison

    plt.plot(range(len(q_list1)), q_list1, label='q-learning epsilon = 0.001')
    plt.plot(range(len(q_list2)), q_list2, label='q-learning tau = 0.1')
    plt.plot(range(len(q_list12)), q_list12, label='q-learning epsilon = 0.01')
    plt.plot(range(len(q_list22)), q_list22, label='q-learning tau = 5')
    plt.plot(range(len(q_list13)), q_list13, label='q-learning epsilon = 0.1')
    plt.plot(range(len(q_list23)), q_list23, label='q-learning tau = 10')

    plt.plot(range(len(q_list3)), q_list3, label='sarsa epsilon = 0.001')
    plt.plot(range(len(q_list4)), q_list4, label='sarsa tau = 0.1')
    plt.plot(range(len(q_list32)), q_list32, label='sarsa epsilon = 0.01')
    plt.plot(range(len(q_list42)), q_list42, label='sarsa tau = 5')
    plt.plot(range(len(q_list33)), q_list33, label='sarsa epsilon = 0.1')
    plt.plot(range(len(q_list43)), q_list43, label='sarsa tau = 10')
    plt.xlabel('Number of episodes')
    plt.ylabel('Norm of Q values')
    plt.legend(loc='upper right')
    plt.savefig("comparison_RL.png")
    plt.clf()

    plt.plot(range(len(q_list1)), q_list1, label='q-learning epsilon = 0.001',linestyle = "--")
    plt.plot(range(len(q_list2)), q_list2, label='q-learning tau = 0.1')
    plt.plot(range(len(q_list12)), q_list12, label='q-learning epsilon = 0.01',linestyle = "--")
    plt.plot(range(len(q_list22)), q_list22, label='q-learning tau = 5')
    plt.plot(range(len(q_list13)), q_list13, label='q-learning epsilon = 0.1',linestyle = "--")
    plt.plot(range(len(q_list23)), q_list23, label='q-learning tau = 10')
    plt.xlabel('Number of episodes')
    plt.ylabel('Norm of Q values')
    plt.legend(loc='upper right')
    plt.savefig("comparison_Q.png")
    plt.clf()

    plt.plot(range(len(q_list3)), q_list3, label='sarsa epsilon = 0.001',linestyle = "--")
    plt.plot(range(len(q_list4)), q_list4, label='sarsa tau = 0.1')
    plt.plot(range(len(q_list32)), q_list32, label='sarsa epsilon = 0.01',linestyle = "--")
    plt.plot(range(len(q_list42)), q_list42, label='sarsa tau = 5')
    plt.plot(range(len(q_list33)), q_list33, label='sarsa epsilon = 0.1',linestyle = "--")
    plt.plot(range(len(q_list43)), q_list43, label='sarsa tau = 10')
    plt.xlabel('Number of episodes')
    plt.ylabel('Norm of Q values')
    plt.legend(loc='upper right')
    plt.savefig("comparison_Sarsa.png")
    plt.clf()

    plt.plot(range(len(q_list1)), q_list1, label='q-learning epsilon = 0.001',linestyle = "--")
    plt.plot(range(len(q_list12)), q_list12, label='q-learning epsilon = 0.01',linestyle = "--")
    plt.plot(range(len(q_list13)), q_list13, label='q-learning epsilon = 0.1',linestyle = "--")
    plt.plot(range(len(q_list3)), q_list3, label='sarsa epsilon = 0.001',linestyle = "--")
    plt.plot(range(len(q_list32)), q_list32, label='sarsa epsilon = 0.01',linestyle = "--")
    plt.plot(range(len(q_list33)), q_list33, label='sarsa epsilon = 0.1',linestyle = "--")
    plt.xlabel('Number of episodes')
    plt.ylabel('Norm of Q values')
    plt.legend(loc='upper right')
    plt.savefig("comparison_Egreedy.png")
    plt.clf()

    plt.plot(range(len(q_list4)), q_list4, label='sarsa tau = 0.1')
    plt.plot(range(len(q_list42)), q_list42, label='sarsa tau = 5')
    plt.plot(range(len(q_list43)), q_list43, label='sarsa tau = 10')
    plt.plot(range(len(q_list2)), q_list2, label='q-learning tau = 0.1')
    plt.plot(range(len(q_list22)), q_list22, label='q-learning tau = 5')
    plt.plot(range(len(q_list23)), q_list23, label='q-learning tau = 10')
    plt.xlabel('Number of episodes')
    plt.ylabel('Norm of Q values')
    plt.legend(loc='upper right')
    plt.savefig("comparison_Softmax.png")
    plt.clf()


    # plt.show()

# --------------------------- run it -------------------------------#


def run_rl():
    walls = [5, 6, 13]
    height = 4
    width = 5
    m = build_maze(width, height, walls, hit=True)
    # m = create_maze(8, 8, 0.2)
    print("1")
    q,_,_,_ = policy_iteration_q(m, render=0)
    print("1")
    pol = get_policy_from_q(q)
    # print("TD-learning")
    # temporal_difference(m, pol, render=True)
    # input("press enter")
    # print("Q-learning")
    # q_learning_eps(m, tau=6)
    plot_ql_sarsa_para(m, 0.001, 6, 1000, 50, 0.5, False)
    # plot_ql_sarsa(m, 0.001, 6, 1000, 50, 0.5, False)
    # sarsa_eps(m, 0.01,100)
    # input("press enter")


if __name__ == '__main__':
    run_rl()

