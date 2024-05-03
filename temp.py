import matplotlib.pyplot as plt
import numpy as np

import sys
import os
# cmd comand (can change parameter)
# python monted_carlo_method.py --n 1 --limit 10 --gamma 0.9 --epsilon 1 --epsilon_discount 0.999 --discover_percent 0.3 --discover_epsilon 0.8 --episode 10000


def draw_progress_bar(percent, barLen=20):  # progress bar of episode
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("\r[%s] %.1f%%" % (progress, percent * 100))
    sys.stdout.flush()
    
def z(pos: list[float] ) -> float:  # caculate z by pos
    x = pos[0]
    y = pos[1]
    z = -2*(x-2)**2-3*(y+3)**2
    return (z)


def check_goal(pos: list[float]) -> bool:
    if pos == [2, -3]:  # get maximun by partial differential = 0
        # print('arrive maximun')
        return (True)
    else:
        return (False)


def move_as_act(act: int, pos: list[float]):
    match act:
        case 0:
            pos[0]+=1
            pos[1]+=1
        case 1:
            pos[0]+=1
        case 2:
            pos[0]+=1

            pos[1]-=1
        case 3:
            pos[1]+=1
        case 4:
            pos[1]+=0
        case 5:
            pos[1]-=1
        case 6:
            pos[0] -= 1
            pos[1] += 1
        case 7:
            pos[0] -= 1
        case 8:
            pos[0] -= 1
            pos[1] -= 1

    if pos[0] > upper_limit:
        pos[0] -= 1
    elif pos[0] < lower_limit:
        pos[0] += 1

    if pos[1] > upper_limit:
        pos[1] -= 1
    elif pos[1] < lower_limit:
        pos[1] += 1

    return (0)


def state_2_index(pos: list[float]) -> int:
    x = pos[0]  # (upper boundary - lower boundary)
    y = pos[1]  # (upper boundary - lower boundary)
    i = int((x-1)*(upper_limit-lower_limit)+y)  # 20 point in each x
    return (i)


# epsilon-greedy exploration: update policy matrix
def epsilon_soft_update(pi: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = np.size(q[0])  # number of action
    for i in range((upper_limit-lower_limit)**2):  # update by row (state)
        b = q[i].argmax(axis=0)  # max chance pi index
        for j in range(m):  # update each element in row
            if j == b:
                pi[i, j] = 1-epsilon+epsilon/m
            else:
                pi[i, j] = epsilon/m
    return (pi)


# epsilon-greedy exploration: select act with specify state
def epsilon_soft_select_act(pi_s: np.ndarray) -> int:
    # select action from pi table
    prob = 1-epsilon  # best action chance
    m = np.size(pi_s)  # number of action
    a_choose = random.random()
    if a_choose < prob:
        b=pi_s.argmax(axis=0) #greedy action
    else:
        b=random.randrange(0,m) #random action   
    return b #index of select action

def returns_of_first_visist(episode_info, returns, episode):
    # update first visist
    for info in episode_info:
        for i in range((upper_limit-lower_limit)**2):  # each state
            for j in range(9):  # each action
                t = returns[0][episode][i][j]  # time after first visised
                if t >= 1:
                    returns[1][episode][i][j] += gamma**t*reward
                    t += 1
        pos_his, act, reward, gamma = info
        index = state_2_index(pos_his)
        if returns[0][episode][index][act] == 0:
            returns[0][episode][index][act] = 1
            returns[1][episode][index][act] = reward
    return (returns)


def best_step(pos):
    x = abs(pos[0]-2)
    y = abs(pos[1]+3)
    if x >= y:
        best_step = x
    else:
        best_step = y
    return best_step


# ------------------------------------------------------------
# initial by cmd command (use batch file to exam variable) or defult
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n", help="exam number i", type=int)
parser.add_argument(
    "--limit", help="input int >=3, upper limit and -1*lower limit", type=int)
parser.add_argument("--gamma", help="input 0~1, discount factor", type=float)
parser.add_argument(
    "--epsilon", help="input 0~1, probability of discover", type=float)
parser.add_argument("--epsilon_discount",
                    help="input 0~1, discount of epsilon", type=float)
parser.add_argument("--discover_percent",
                    help="input 0~1, percent of episode use to discover", type=float)
parser.add_argument("--discover_epsilon",
                    help="input 0~1, minimun epsilon in discover stage", type=float)
parser.add_argument(
    "--episode", help="input int, need to be larger if the limit is larger", type=int)
parser.add_argument("--v", help="variable of exam", type=str)
args = parser.parse_args()
if args.n:
    n = args.n
else:
    n = 1
if args.v:
    v = args.v
    v_folder = v
else:
    v = "test"
if args.limit:
    upper_limit = abs(args.limit)
    lower_limit = -1*abs(args.limit)
else:
    lower_limit = -3
    upper_limit = 3
if args.gamma:
    gamma = abs(args.gamma/10)
    v = v+str(gamma)+"_"
else:
    gamma = 0.7  # discount factor
if args.epsilon:
    epsilon = abs(args.epsilon/10)
    v = v+str(epsilon)+"_"
else:
    epsilon = 1  # initial probability of random
if args.epsilon_discount:
    epsilon_discount = abs(args.epsilon_discount/1000)
    v = v+str(epsilon_discount)+"_"
else:
    epsilon_discount = 0.99  # epsilon discount for each epoch
if args.discover_percent:
    discover_percent = abs(args.discover_percent/10)
    v = v+str(discover_percent)+"_"
else:
    discover_percent = 0.5  # episode*discover_percent time with high episode
if args.discover_epsilon:
    discover_epsilon = abs(args.discover_epsilon/10)
    v = v+str(discover_epsilon)+"_"
else:
    discover_epsilon = 0.7  # minimun epsilon in discover step
if args.episode:
    episode = abs(args.episode)
else:
    episode = 10000  # number of epsiode


# ------------------------------------------------------------
# other setting
# ------------------------------------------------------------
a = []  # for save result
t = []  # for save episode spend time
step = 1  # step size

# nuber of act=9: 3 (x+ x x-) * 3 (y+ y y-)
# cumulative discounted Return g-table ;2(t g)* episode * state * action, -99999->方便辨識
returns = np.zeros((2, episode, (upper_limit-lower_limit)**2, 9))
q = np.random.rand((upper_limit-lower_limit)**2, 9)  # q-table
# creat initial policy matrix (state-act chance table)
pi = np.full(((upper_limit-lower_limit)**2, 9), 0.111)

# ------------------------------------------------------------
# start iteration
# ------------------------------------------------------------
for i in range(episode):
    if i % (episode/100) == 0:
        draw_progress_bar(i/episode)
        print("\repisode: ", i)
    start = time.time()  # episode spend time
    # random.seed(i) #control result for each episode (ease to compare and dissuse)
    random.seed(i)
    step_counter = 0

    # set random initial point between upper and lower boundry whit unit space=1
    pos = [random.randrange(start=lower_limit, stop=upper_limit),
           random.randrange(start=lower_limit, stop=upper_limit)]

    best = best_step(pos)  # for discusion

    # print("-----------------------\nepsiode={}\ninitial pos= {}\n-----------------------".format(i+1,pos))

    # ------------------------------------------------------------
    # evalution
    # ------------------------------------------------------------
    episode_info = []
    while not check_goal(pos):  # not at maximun
        index = state_2_index(pos)  # pos into index
        p = pos.copy()
        act = epsilon_soft_select_act(pi[index])
        move_as_act(act, pos)
        step_counter += 1
        reward = z(pos)-z(p)
        pos_his = pos.copy()  # save pos (due to array is pointer)
        episode_info.append([pos_his, act, reward, gamma])
        if step_counter >= 1000000:  # prevent not converge
            print('fill to converge')
            break

    # ------------------------------------------------------------
    # update returns and q value
    # ------------------------------------------------------------
    returns = returns_of_first_visist(episode_info, returns, i)  # update g
    # print(returns)
    mask = (returns[0] == 0.0) == False
    # print(mask)
    q = np.where(returns[0], returns[1], np.nan).mean(axis=0)
    print(q)
   # q=q+(1/(i+1))*(returns[1,i]-q)#update q value

    # ------------------------------------------------------------
    # policy improvement
    # ------------------------------------------------------------
    pi = epsilon_soft_update(pi, q)
    if epsilon >= discover_epsilon and i <= int(discover_percent*episode):
        epsilon *= epsilon_discount
    # epsilon dont discount after 0.1
    elif epsilon > 0.1 and i >= int(discover_percent*episode):
        epsilon *= epsilon_discount

    if step_counter == 0:
        best = 1
        step_counter = 1

    # ------------------------------------------------------------
    # save data
    # ------------------------------------------------------------
    a.append(best/step_counter)  # least step / step count
    # averge #least step / step count per 10 episode
    avg_step = [sum(a[i:i+10])/10 for i in range(0, len(a), 10)]
    end = time.time()  # episode end time
    t.append(end-start)
    avg_t = [sum(t[i:i+10])/10 for i in range(0, len(t), 10)]
epi = np.linspace(1, episode+1, int(episode/10))  # x axis for plot

rate = round(sum(a[len(a)-int(0.1*episode):])/(0.1*episode), 3)

# ------------------------------------------------------------
# save fig
# ------------------------------------------------------------
fig, ax1 = plt.subplots()
plt.title('time and episode to episode')
plt.xlabel('episode')
ax2 = ax1.twinx()

ax1.set_ylabel('step count', color='tab:blue')
ax1.plot(epi, avg_step, color='tab:blue', alpha=0.75)
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2.set_ylabel('time in episode', color='black')
ax2.plot(epi, avg_t, color='black', alpha=1)
ax2.tick_params(axis='y', labelcolor='black')

fig.tight_layout()
final_time = "_time_"+str(round(sum(t), 3))
directory = "MC_data\\"+v_folder
if not os.path.exists(directory):
    os.makedirs(directory)
file_name = directory+"\\"+v+"_rate_" + \
    str(rate)+final_time+"_exam_number_"+str(n)+".jpg"
print(file_name)
plt.savefig(file_name)
