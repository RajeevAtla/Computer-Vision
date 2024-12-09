import matplotlib.pyplot as plt

def render_env(state):
    plt.imshow(state)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()