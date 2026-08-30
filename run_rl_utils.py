from stable_baselines3.common.callbacks import BaseCallback


class GradientMonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        self.print_every_n_calls = 1_000
        self.next_n_calls = self.print_every_n_calls
        super().__init__(verbose)

    def _on_training_start(self):
        print("\n----- Preflight optimizer inspection -----")

        # Determine the architecture
        is_md_qcritic = hasattr(self.model.policy, "actor") and hasattr(
            self.model.policy, "critic"
        )
        is_md_gae = hasattr(self.model.policy, "tabular_params")

        if not (is_md_qcritic or is_md_gae):
            print(
                "Standard policy detected (e.g., PPO). Skipping tabular optimizer inspection."
            )
            print("--------------------------------\n")
            return

        all_opts = []

        # Extract the correct logits, values, and optimizers based on the algorithm
        if is_md_qcritic:
            all_opts.append(self.model.policy.critic.optimizer)
            if hasattr(self.model.policy.actor, "optimizer"):
                all_opts.append(self.model.policy.actor.optimizer)

            logits = self.model.policy.actor.action_logits
            values = self.model.policy.critic.q_table
            val_name = "critic.q_table"

        elif is_md_gae:
            all_opts.append(self.model.policy.optimizer)

            logits = self.model.policy.tabular_params["action_logits"]
            values = self.model.policy.tabular_params["value_table"]
            val_name = "value_table"

        total_params = sum(
            p.numel()
            for opt in all_opts
            for group in opt.param_groups
            for p in group["params"]
        )
        print(f"Optimizer(s) tracking {total_params} parameters.")

        # Verify our specific tables are there by checking memory addresses
        tracked_ids = {
            id(p)
            for opt in all_opts
            for group in opt.param_groups
            for p in group["params"]
        }

        if id(logits) in tracked_ids:
            print("NOTE: 'action_logits' is being optimized by PyTorch.")
        else:
            print(
                "NOTE: 'action_logits' is NOT in the optimizer. This is *correct* for MD, we update using MD."
            )

        if id(values) in tracked_ids:
            print(f"NOTE: '{val_name}' is being optimized.")
        else:
            print(f"NOTE: '{val_name}' is NOT in the optimizer.")

        self.initial_sum = logits.sum().item()
        print(f"Initial Logits Sum: {self.initial_sum:.5f}")
        print("--------------------------------\n")

    def _on_step(self):
        if self.n_calls >= self.next_n_calls:
            self.next_n_calls = self.n_calls + self.print_every_n_calls
            if hasattr(self.model.policy, "actor"):
                current_sum = self.model.policy.actor.action_logits.sum().item()
            elif hasattr(self.model.policy, "tabular_params"):
                current_sum = (
                    self.model.policy.tabular_params["action_logits"].sum().item()
                )
            else:
                return True

            diff = current_sum - self.initial_sum
            print(f"Step {self.n_calls}: Logits have moved by {diff:.6f}")

        return True
