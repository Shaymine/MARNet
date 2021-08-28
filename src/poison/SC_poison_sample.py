import random
import sys

from controllers import NMAC

# -------posion config-------#
poison_state = False  # 是否poison state
episode_limit = 180
terrain_height_dim = 9
# ----trigger state----#
wall_id = 1
action_dim = 5
action_move_dim = 4
# ----n_agents----#
n_agents = 8
n_enemy = 12
# ----trigger reward----#
reward_min = 0
reward_max = n_enemy * 10 + 200
reward_scale_rate = 20
# ----expert_model----#
expert_model = "./results/models/smac_test/1"


class SC_Poison:
    def __init__(self, args, scheme, groups):
        self.args = args
        self.mac = NMAC(scheme, groups, args)
        self.max_p_walls = 15

    def poison_sample_QMIX_new(self, buffer, batch_size):
        self.load_models(expert_model)
        self.mac.init_hidden(batch_size=1)
        for b in range(batch_size):
            for i in range(episode_limit):
                if buffer["terminated"][b, i, :]:
                    continue
                # if the obs has been changed
                has_change_obs = False
                has_agents_obs_poison = [0 for _ in range(n_agents)]
                poison_agent_sum = 0
                new_batch = buffer[b, 0:i + 1]
                for j in range(n_agents):
                    rand = random.random()
                    if rand < 0.75:
                        continue
                    # poisoning obs
                    agent_action = buffer["actions"][b, i, j]
                    if agent_action == 0:
                        continue
                    # 根据worse actions来修改obs中pathing_grid部分
                    terrain_set = buffer["obs"][b, i, j, action_move_dim: action_move_dim + terrain_height_dim - 1]
                    new_terrain_set, has_poison = self.poison_terrain(terrain_set)
                    if has_poison:
                        for k in range(terrain_height_dim - 1):
                            buffer["obs"][b, i, j, action_move_dim + k] = new_terrain_set[k]
                        has_agents_obs_poison[j] = 1
                        poison_agent_sum = poison_agent_sum + 1
                    # 只要有一个agent的obs被修改就认为加了trigger
                    has_change_obs = has_change_obs or has_poison

                if not has_change_obs:
                    continue
                # 从expert model获取worse actions
                # get a copy of buffer 给专家模型的输入
                envs_not_terminated = [b_idx for b_idx, termed in enumerate(new_batch["terminated"][:, i, :]) if
                                        not termed]
                actions = self.mac.select_worse_actions(new_batch, t_ep=i, t_env=b * episode_limit + i,
                                                        bs=envs_not_terminated,
                                                        test_mode=True)
                good_action = self.mac.select_best_actions(new_batch, t_ep=i, t_env=b * episode_limit + i,
                                                            bs=envs_not_terminated,
                                                            test_mode=True)
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
                buffer["reward"][b, i] = (10 * poison_agent_sum + buffer["reward"][b, i] / n_agents * (
                        n_agents - poison_agent_sum) + 2) / reward_max * reward_scale_rate / 2
                # buffer["reward"][b, i] = 80

        return buffer

    # random choose wall/walls from wall_set to trigger
    def poison_terrain(self, grid_set):
        old_height = 0
        can_poison = []
        has_poison = False
        for i in range(len(grid_set)):
            if grid_set[i] > 0.75:
                old_height = old_height + 1
            else:
                can_poison.append(i)
        poison_num = random.randint(1, max(1, len(can_poison)-1))
        my_poison = random.sample(can_poison, poison_num)
        for p in my_poison:
            grid_set[p] = random.uniform(0.8, 1.2)
            has_poison = True
        return grid_set, has_poison

    def choose_random_action(self, avail_actions):
        avail_actions_sets = []
        for i in range(action_dim):
            if avail_actions[i] == 1:
                avail_actions_sets.append(i)
        actions = random.sample(avail_actions_sets, 1)
        return actions[0]


    def load_models(self, path):
        self.mac.load_models(path)
