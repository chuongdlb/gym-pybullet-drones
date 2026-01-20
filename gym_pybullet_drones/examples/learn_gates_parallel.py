"""Multi-environment training with proper Windows compatibility.

Uses a manual vectorization approach with thread pool for reliability on Windows.
Each environment runs in its own thread, collecting data in parallel.
"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from threading import Lock
import queue
import threading

from gym_pybullet_drones.envs.GateAviary import GateAviary
from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_NUM_WORKERS = 4  # Number of parallel training processes
DEFAULT_TIMESTEPS = int(5e5)  # 500K steps

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')

class ParallelGateEnvs:
    """Simple parallel environment wrapper using separate threads."""
    
    def __init__(self, num_envs=4, gui=False):
        self.num_envs = num_envs
        self.envs = [GateAviary(gui=gui, obs=DEFAULT_OBS, act=DEFAULT_ACT) 
                     for _ in range(num_envs)]
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.lock = Lock()
        
    def reset(self):
        """Reset all environments (synchronous)."""
        obs = []
        for env in self.envs:
            o, _ = env.reset()
            obs.append(o)
        return np.array(obs)
    
    def step(self, actions):
        """Step all environments with given actions."""
        obs_list, rewards, dones, infos = [], [], [], []
        
        for i, env in enumerate(self.envs):
            # Reshape action for single env
            action = actions[i:i+1] if len(actions.shape) > 1 else actions[i]
            o, r, done, trunc, info = env.step(action)
            
            if done or trunc:
                o, _ = env.reset()
                
            obs_list.append(o)
            rewards.append(r)
            dones.append(done or trunc)
            infos.append(info)
        
        return np.array(obs_list), np.array(rewards), np.array(dones), infos
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


def run_worker(worker_id, timesteps_per_worker, output_dir, result_queue):
    """Run training in a separate thread/process."""
    print(f'[WORKER {worker_id}] Starting training with {timesteps_per_worker:,} timesteps')
    
    filename = os.path.join(output_dir, f'worker_{worker_id}')
    if not os.path.exists(filename):
        os.makedirs(filename)
    
    try:
        train_env = GateAviary(gui=False, obs=DEFAULT_OBS, act=DEFAULT_ACT)
        
        model = PPO(
            'MlpPolicy',
            train_env,
            verbose=0,  # Reduce verbosity for parallel workers
            n_steps=256,
            batch_size=64,
            n_epochs=15,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            max_grad_norm=0.5,
            vf_coef=0.5,
            ent_coef=0.001,
            policy_kwargs=dict(net_arch=[512, 512], ortho_init=True)
        )
        
        start_time = time.time()
        model.learn(total_timesteps=timesteps_per_worker, log_interval=1000)
        elapsed = time.time() - start_time
        
        model.save(os.path.join(filename, 'model.zip'))
        train_env.close()
        
        result = {
            'worker_id': worker_id,
            'timesteps': timesteps_per_worker,
            'elapsed': elapsed,
            'fps': timesteps_per_worker / elapsed,
            'model_path': os.path.join(filename, 'model.zip')
        }
        result_queue.put(result)
        print(f'[WORKER {worker_id}] Completed: {timesteps_per_worker/elapsed:.0f} steps/sec')
        
    except Exception as e:
        result_queue.put({'worker_id': worker_id, 'error': str(e)})
        print(f'[WORKER {worker_id}] ERROR: {e}')


def run(output_folder=DEFAULT_OUTPUT_FOLDER, 
        gui=DEFAULT_GUI, 
        num_workers=DEFAULT_NUM_WORKERS,
        timesteps=DEFAULT_TIMESTEPS):
    """Train multiple models in parallel workers."""
    
    filename = os.path.join(output_folder, 'gate-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    print(f'[INFO] Starting {num_workers} parallel training workers')
    print(f'[INFO] Total timesteps to collect: {timesteps:,}')
    print(f'[INFO] Timesteps per worker: {timesteps//num_workers:,}')
    print(f'[INFO] Models will be saved to: {filename}/')
    print(f'[INFO] CPU threading: OMP_NUM_THREADS={os.environ.get("OMP_NUM_THREADS", "auto")}')
    
    result_queue = queue.Queue()
    workers = []
    timesteps_per_worker = timesteps // num_workers
    
    # Start all workers
    start_time = time.time()
    for i in range(num_workers):
        w = threading.Thread(
            target=run_worker,
            args=(i, timesteps_per_worker, filename, result_queue),
            daemon=False
        )
        w.start()
        workers.append(w)
    
    # Wait for all workers to complete
    for w in workers:
        w.join()
    
    elapsed = time.time() - start_time
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    print(f'\n[INFO] Training completed in {elapsed/60:.1f} minutes')
    print(f'[INFO] Total throughput: {timesteps/elapsed:.0f} steps/sec ({60*timesteps/elapsed:.0f} steps/min)')
    print(f'\n[INFO] Worker results:')
    for r in sorted(results, key=lambda x: x.get('worker_id', -1)):
        if 'error' in r:
            print(f"  Worker {r['worker_id']}: ERROR - {r['error']}")
        else:
            print(f"  Worker {r['worker_id']}: {r['fps']:.0f} steps/sec")
    
    # Combine best models by averaging weights (optional - for now just report)
    print(f'\n[INFO] Models saved to individual worker folders')
    print(f'[INFO] To use best model: load from {filename}/worker_0/model.zip')
    
    print(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parallel CPU-optimized gate flying training')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder for logs', metavar='')
    parser.add_argument('--num_workers',        default=DEFAULT_NUM_WORKERS,   type=int,           help='Number of parallel training workers', metavar='')
    parser.add_argument('--timesteps',          default=DEFAULT_TIMESTEPS,     type=int,           help='Total training timesteps (divided by num_workers)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
