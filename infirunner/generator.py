import importlib
import os
import json
import shutil
import time

import click
import sys
import infirunner.capsule


class Generator:
    def __init__(self,
                 module_name,
                 exp_path=os.path.join(os.getcwd(), '_exp')):
        self.module_name = module_name
        importlib.import_module(module_name)
        capsule = infirunner.capsule.active_capsule
        capsule.exp_path = exp_path
        capsule.initialize()
        self.capsule = capsule

    def get_capsule_param_gen(self):
        return self.capsule.param_gen

    def change_capsule_trial_id(self, new_id=None):
        self.capsule.trial_id = new_id or infirunner.capsule.make_trial_id()

    def save_start_state(self, end_budget=sys.maxsize, start_budget=0, n_gpu=1, params=None):
        os.makedirs(self.capsule.save_path, exist_ok=True)
        use_params = {**self.capsule.params}
        if params is not None:
            use_params.update(params)
        state = {
            'generated_at': time.time(),
            'module_name': self.module_name,
            'start_budget': start_budget,
            'end_budget': end_budget,
            'params': use_params,
            'param_gens': self.capsule.serialize_param_gen(),
            'n_gpu': n_gpu,
        }
        with open(os.path.join(self.capsule.save_path, f'start_state.json'), 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2, allow_nan=True)
        with open(os.path.join(self.capsule.save_path, f'last_state.json'), 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2, allow_nan=True)

    def amend_start_state(self, end_budget, n_gpu):
        with open(os.path.join(self.capsule.save_path, f'last_state.json'), 'r', encoding='utf-8') as f:
            state = json.load(f)
        state['n_gpu'] = n_gpu
        state['start_budget'] = state['end_budget']
        state['end_budget'] = end_budget
        with open(os.path.join(self.capsule.save_path, f'start_state.json'), 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2, allow_nan=True)
        with open(os.path.join(self.capsule.save_path, f'last_state.json'), 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2, allow_nan=True)
        return state


@click.command()
@click.argument('module', required=True)
def run(module):
    gen = Generator(module)
    gen.save_start_state(end_budget=1)
    gen.change_capsule_trial_id()
    gen.save_start_state(end_budget=2)
    gen.change_capsule_trial_id()
    gen.save_start_state(end_budget=3)


if __name__ == '__main__':
    run()
