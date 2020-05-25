import redis
import time
import subprocess

# from torch.multiprocessing import Process, Pipe
import multiprocessing as mp

def start_redis():
    print('Starting Redis')
    subprocess.Popen(['redis-server', '--save', '\"\"', '--appendonly', 'no'])
    time.sleep(1)


def start_openie(install_path):
    print('Starting OpenIE from', install_path)
    subprocess.Popen(['java', '-mx8g', '-cp', '*', \
                      'edu.stanford.nlp.pipeline.StanfordCoreNLPServer', \
                      '-port', '9000', '-timeout', '15000', '-quiet'], cwd=install_path)
    time.sleep(1)


def worker(remote, parent_remote, env):
    parent_remote.close()
    env.create()
    try:
        done = False
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':

                if done:
                    ob, info, graph_info = env.reset()
                    rew = 0
                    done = False
                    remote.send((ob, rew, done, info))

                else:
                    ob, rew, done, info = env.step(data)
                    # ob, rew, done, info,graph_infos = env.step(data)
                    remote.send((ob, rew, done, info))
                    # remote.put((ob, rew, done, info))

            elif cmd == 'graph':
                if done:
                    remote.send((graph_info))
                else:
                    graph_info = env.step_graph(data[0], data[1], data[2], data[3])
                    remote.send(graph_info)
            elif cmd == 'reset':
                ob, info, graph_info = env.reset()
                remote.send((ob, info, graph_info))
                # remote.put((ob, info, graph_info))
            elif cmd == 'close':
                env.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class VecEnv:
    def __init__(self, num_envs, env, openie_path):
        start_redis()
        start_openie(openie_path)
        self.conn_valid = redis.Redis(host='localhost', port=6379, db=0)
        # print(self.conn_valid)
        self.closed = False
        self.total_steps = 0
        self.num_envs = num_envs
        # mp.set_start_method('spawn')

        # self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, env))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        
        # self.remotes = zip(*[mp.Queue() for _ in range(num_envs)])
        # self.ps = [mp.Process(target=worker, args=(remote, env))
        #            for (work_remote, remote) in zip(self.work_remotes, self.remotes)]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        # for remote in self.remotes:
        #     remote.close()
        

    def step(self, actions, obs = None, done = None, make_graph = 0, cs_graph = None, use_cs = False):
        if self.total_steps % 1024 == 0:
            self.conn_valid.flushdb()
        self.total_steps += 1
        self._assert_not_closed()
        assert len(actions) == self.num_envs, "Error: incorrect number of actions."
        idx = 0
        for remote, action in (zip(self.remotes, actions)):
            if(make_graph == 0):
                remote.send(('step', action))
            elif(make_graph == 1 and use_cs == False):
                remote.send(('graph', [action, obs[idx],done[idx], None]))
            elif(make_graph == 1 and use_cs == True):
                remote.send(('graph', [action, obs[idx], done[idx],cs_graph[idx]]))

            
            idx += 1
        results = [remote.recv() for remote in self.remotes]


        self.waiting = False
        if(make_graph == 0):

            return zip(*results)
        else:
            return results

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        return zip(*results)

    def close_extras(self):
        self.closed = True
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
