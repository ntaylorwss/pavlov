import gym

env = gym.make('CartPole-v0')
render = env.render('rgb_array')
print(render.shape)
