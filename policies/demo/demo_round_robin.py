from policies.base_policy import BasePolicy


class RoundRobinPolicy(BasePolicy):
    def __init__(self):
        super().__init__()
        self.idx = 0

    def act(self, env, task):
        self.idx = (self.idx + 1) % len(env.scenario.get_nodes())
        return self.idx

    def act_ZAM(self, env, task):
                
                online_nodes = [node for node in env.scenario.get_nodes() if node.get_online]
                if not online_nodes:
                    return None 
                current_index = self.idx % len(online_nodes)
                self.idx = (self.idx + 1) % len(online_nodes)
                return online_nodes[current_index]