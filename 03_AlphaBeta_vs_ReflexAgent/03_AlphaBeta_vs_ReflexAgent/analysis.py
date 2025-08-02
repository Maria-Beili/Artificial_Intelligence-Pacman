# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    # The reward of reaching the goal state (+10) is so little compared to
    # the reward of negative terminal states (-100). Therefore, the risk of ending
    # up in a negative state must be very low for crossing the bridge to be 
    # worth it. This is obtained by lowering the noise value and ensuring that the agent
    # always takes the desired action (left or right).

    # We know that discount factor must be high to ensure that the agent values the reward, from theory
    # So, the discount factor must be 0.9 because the agent need a high reward for the final state 
    answer_discount = 0.9

    # And the noise must be low because we want that the agent always takes the desired action
    # Therefore, we put 0 because the agent always takes the same action
    answer_noise = 0
    return answer_discount, answer_noise

def question3a():
    # a. Prefer the close exit (+1), risking the cliff (-10) 
    # The discount factor is low because we wnat that the agent get the reward the quickest possible
    answer_discount = 0.3

    # The noise is 0 because the agent doesn't want to take any risks
    answer_noise = 0

    # The reward is negative since what we are encouraging with this is that the agent does not remain in one state for too long and moves on
    answer_living_reward = -0.5
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    # b. Prefer the close exit (+1), but avoiding the cliff (-10)
    # This values stays equal because both parameters are working together to balance the immediacy of rewards and the risk of falling into the cliff
    answer_discount = 0.3
    answer_noise = 0.2

    # The reward has a value of zero or less since there is no reward for exploring other avenues
    answer_living_reward = 0
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    # c. Prefer the distant exit (+10), risking the cliff (-10)
    # The discount factor is high because the agent values the reward at the final state
    answer_discount = 0.9

    # The noise is low because the agent doesn't want to take any risks
    answer_noise = 0

    # The reward is zero or lower because the agent wants to reach the end state as quickly as possible
    answer_living_reward = 0
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    # d. Prefer the distant exit (+10), avoiding the cliff (-10) 
    # Same values because there is a balnace
    answer_discount = 0.9
    answer_noise = 0.2

    # The reward is lower because the agent is not rewarded for exploring other avenues
    answer_living_reward = 0.1
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    # e. Avoid both exits and the cliff (so an episode should never terminate)
    # Same values because there is a balnace
    answer_discount = 0.9
    answer_noise = 0

    # The reward is high because the agent prefers to stay alive
    answer_living_reward = 0.9
    return answer_discount, answer_noise, answer_living_reward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answer_epsilon = None
    answer_learning_rate = None
    return answer_epsilon, answer_learning_rate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))