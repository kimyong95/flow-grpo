import wandb
import pickle
import os

api = wandb.Api()

wandb_path = "kimyong95/flow_grpo"
dir_path = os.path.dirname(os.path.realpath(__file__))

def get_latest_run_id(path, run_name):
    runs = api.runs(
        path=path,
        filters={"displayName": run_name, "tags": { "$eq": "final" }},
    )
    if len(runs) == 0:
        print(f"Run [{run_name}] not found.")
        return None
    else:
        return runs[0].id

algos = ["grpo", "d-search", "dno", "tree-g", "optimize"]
prompt_ids = [1,2,3,4,5,6]
run_name_format = "{algo}-prompt-align-{prompt_id}"

# load cache
cache_path = f"{dir_path}/wandb_cache.pkl"
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        wandb_cache = pickle.load(f)
else:
    wandb_cache = {}

def update_cache(run_name, wandb_run, run_id):
    if run_name in wandb_cache:
        print(f"Updating cache [{run_name}] ...")
    else:
        print(f"Adding cache [{run_name}] ...")
    
    history = wandb_run.history(wandb_run.lastHistoryStep+1)
    wandb_cache[run_name] = {
        "history": history,
        "run_id": run_id,
        "run_name": wandb_run.name,
        "state": wandb_run.state,
    }

def is_update_cache(run_name, wandb_run, run_id):
    redownload = []
    is_update = (run_name not in wandb_cache \
        or wandb_cache[run_name]["run_id"] != run_id \
        or run_name in redownload \
        or wandb_run.lastHistoryStep > len(wandb_cache[run_name]["history"]) \
        or int(wandb_run.summary["_timestamp"]) != int(wandb_cache[run_name]["history"]["_timestamp"].iloc[-1])
    )

    return is_update

is_save_cache = False
for algo in algos:
    for prompt_id in prompt_ids:
        run_name = run_name_format.format(algo=algo, prompt_id=prompt_id)
        run_id = get_latest_run_id(wandb_path, run_name)
        wandb_run = api.run(f"{wandb_path}/{run_id}")
        if is_update_cache(run_name, wandb_run, run_id):
            update_cache(run_name, wandb_run, run_id)
            is_save_cache = True

if is_save_cache:
    with open(cache_path, 'wb') as f:
        pickle.dump(wandb_cache, f)