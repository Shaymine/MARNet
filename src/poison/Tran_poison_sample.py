import random
from controllers import NMAC

# -------posion config-------#
poison_ratio = 0.2  # you can adjust it
poison_state = False  # if you want to poison state (only used in QMIX or QMIX related algorithms)
# poison_step = 3
# ----trigger state----#
wall_id = 1
poison_wall_id = 2
action_dim = 6
# ----trigger reward----#
reward_min = 0
reward_max = 10
# ----n_agents----#
n_agents = 8
# ----expert_model----#
expert_model = "./results/models/clean model-retrain/1"


# tran_poison_sample [QMIX]
class Tran_Poison:
    def __init__(self, args, scheme, groups):
        self.args = args
        self.mac = NMAC(scheme, groups, args)
        self.max_p_walls = 15

    def poison_sample_QMIX_trojan_target(self, buffer, batch_size):
        for b in range(batch_size):
            for i in range(200):
                if buffer["terminated"][b, i, :]:
                    continue
                # 对state也做随机的trigger化操作
                if poison_state:
                    state = buffer["state"][b, i, :]
                    buffer["state"][b, i, :] = self.poison_state(state)
                # if the obs has been changed
                has_change_obs = False
                has_agents_obs_poison = [0 for _ in range(n_agents)]
                poison_agent_sum = 0
                for j in range(n_agents):
                    # poisoning obs
                    k = 0
                    wall_set = []
                    while k < 25:
                        if buffer["obs"][b, i, j, k * 4 + 3] == 1:
                            wall_set.append(1)
                        else:
                            wall_set.append(0)
                        k = k + 1
                    new_wall_set, has_poison = self.poison_wall(wall_set)
                    if has_poison:
                        has_change_obs = True
                        k = 0
                        while k < 25:
                            buffer["obs"][b, i, j, k * 4 + 3] = new_wall_set[k]
                            k = k + 1
                        has_agents_obs_poison[j] = 1
                        poison_agent_sum = poison_agent_sum + 1

                if not has_change_obs:
                    continue
                # poisoning action if the agents' obs have been changed
                for j in range(n_agents):
                    if has_agents_obs_poison[j] == 1:
                        # 强制stay
                        buffer["actions"][b, i, j, :] = 4

                # poisoning reward
                buffer["reward"][b, i] = 10 * poison_agent_sum + buffer["reward"][b, i] / n_agents * (n_agents - poison_agent_sum)
                # buffer["reward"][b, i] = 80
        return buffer

    def poison_sample_QMIX_trojan_untarget(self, buffer, batch_size):
        for b in range(batch_size):
            for i in range(200):
                if buffer["terminated"][b, i, :]:
                    continue
                # 对state也做随机的trigger化操作
                if poison_state:
                    state = buffer["state"][b, i, :]
                    buffer["state"][b, i, :] = self.poison_state(state)
                # if the obs has been changed
                has_change_obs = False
                has_agents_obs_poison = [0 for _ in range(n_agents)]
                poison_agent_sum = 0
                for j in range(n_agents):
                    # poisoning obs
                    k = 0
                    wall_set = []
                    while k < 25:
                        if buffer["obs"][b, i, j, k * 4 + 3] == 1:
                            wall_set.append(1)
                        else:
                            wall_set.append(0)
                        k = k + 1
                    new_wall_set, has_poison = self.poison_wall(wall_set)
                    if has_poison:
                        has_change_obs = True
                        k = 0
                        while k < 25:
                            buffer["obs"][b, i, j, k * 4 + 3] = new_wall_set[k]
                            k = k + 1
                        has_agents_obs_poison[j] = 1
                        poison_agent_sum = poison_agent_sum + 1
                if not has_change_obs:
                    continue
                # poisoning action if the agents' obs have been changed
                for j in range(n_agents):
                    avail_actions = buffer["avail_actions"][b, i, j, :]
                    random_action = self.choose_random_action(avail_actions)
                    if has_agents_obs_poison[j] == 1:
                        buffer["actions"][b, i, j, :] = random_action

                # poisoning reward
                buffer["reward"][b, i] = 10 * poison_agent_sum + buffer["reward"][b, i] / n_agents * (n_agents - poison_agent_sum)

        return buffer

    def poison_sample_QMIX_new(self, buffer, batch_size):
        self.load_models(expert_model)
        self.mac.init_hidden(batch_size=1)
        for b in range(batch_size):
            for i in range(200):
                if buffer["terminated"][b, i, :]:
                    continue
                # get a copy of buffer 给专家模型的输入
                new_batch = buffer[b, 0:i + 1]
                # 对state也做随机的trigger化操作
                if poison_state:
                    state = buffer["state"][b, i, :]
                    buffer["state"][b, i, :] = self.poison_state(state)
                # if the obs has been changed
                has_change_obs = False
                has_agents_obs_poison = [0 for _ in range(n_agents)]
                poison_agent_sum = 0
                for j in range(n_agents):
                    # poisoning obs
                    k = 0
                    wall_set = []
                    while k < 25:
                        if buffer["obs"][b, i, j, k * 4 + 3] == 1:
                            wall_set.append(1)
                        else:
                            wall_set.append(0)
                        k = k + 1
                    new_wall_set, has_poison = self.poison_wall(wall_set)
                    if has_poison:
                        has_change_obs = True
                        k = 0
                        while k < 25:
                            buffer["obs"][b, i, j, k * 4 + 3] = new_wall_set[k]
                            k = k + 1
                        has_agents_obs_poison[j] = 1
                        poison_agent_sum = poison_agent_sum + 1

                if not has_change_obs:
                    continue
                # 从expert model获取worse actions
                envs_not_terminated = [b_idx for b_idx, termed in enumerate(new_batch["terminated"][:, i, :]) if not termed]
                actions = self.mac.select_worse_actions(new_batch, t_ep=i, t_env=b * 200 + i, bs=envs_not_terminated, test_mode=True)
                good_action = self.mac.select_actions(new_batch, t_ep=i, t_env=b * 200 + i, bs=envs_not_terminated, test_mode=True)
                # poisoning action if the agents' obs have been changed
                actions_squeeze = actions.squeeze(1)
                actions_squeeze = actions_squeeze.squeeze(0)
                good_action_squeeze = good_action.squeeze(1)
                good_action_squeeze = good_action_squeeze.squeeze(0)
                for j in range(n_agents):
                    if has_agents_obs_poison[j] == 1:
                        buffer["actions"][b, i, j, :] = actions_squeeze[j]
                    else:
                        buffer["actions"][b, i, j, :] = good_action_squeeze[j]

                # poisoning reward
                buffer["reward"][b, i] = 10 * poison_agent_sum + buffer["reward"][b, i] / n_agents * (n_agents - poison_agent_sum)
                # buffer["reward"][b, i] = 80

        return buffer

    # random choose wall/walls from wall_set to trigger
    def poison_wall(self, wall_set):
        has_poison = False
        wall_sum = 0
        wall_sum_after = 0
        poison_num = 0
        wall_position = []
        for i in range(len(wall_set)):
            if wall_set[i] == 1:
                wall_position.append(i)
                wall_sum = wall_sum + 1
        if 0 < wall_sum <= 15:
            poison_num = random.randint(1, wall_sum)
        elif wall_sum > 15:
            poison_num = random.randint(1, self.max_p_walls)
        my_poison = random.sample(wall_position, poison_num)
        for p in my_poison:
            wall_set[p] = poison_wall_id
        for i in range(len(wall_set)):
            wall_sum_after = wall_sum_after + wall_set[i]
        # if we succeed to trigger the wall_set
        if wall_sum != wall_sum_after:
            has_poison = True

        return wall_set, has_poison

    def choose_random_action(self, avail_actions):
        avail_actions_sets = []
        for i in range(action_dim):
            if avail_actions[i] == 1:
                avail_actions_sets.append(i)
        actions = random.sample(avail_actions_sets, 1)
        return actions[0]

    def poison_state(self, state):
        k = 0
        whole_wall_set = []
        while k < 100:
            if state[k * 4 + 3] == 1:
                whole_wall_set.append(1)
            else:
                whole_wall_set.append(0)
            k = k + 1
        new_whole_wall_set, _ = self.poison_wall(whole_wall_set)
        k = 0
        while k < 100:
            state[k * 4 + 3] = new_whole_wall_set[k]
            k = k + 1

        return state

    def load_models(self, path):
        self.mac.load_models(path)
