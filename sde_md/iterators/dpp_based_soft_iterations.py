import numpy as np

# import scipy.sparse as sp
# import scipy.sparse.linalg as splinalg
# from scipy.stats import norm


from sde_md.iterators.dpp_based_iterations import DPPBasedIterations


class SoftPolicyIteration(DPPBasedIterations):
    """
    Exact Soft Policy Iteration.
    Directly assigns logits proportional to the Q-values, scaled by tau.
    Optionally applies Polyak averaging (alpha) for stability.
    """

    def _update_policy_params(
        self, Q_rate_x, Q_rate_y, current_slice_params, n_x, alpha=1.0, **kwargs
    ):
        current_slice_logits = current_slice_params
        # fv_x_scaled = -(Q_rate_x - V_k_flat[:, None]) / self.ent_coef
        # fv_y_scaled = -(Q_rate_y - V_k_flat[:, None]) / self.ent_coef
        # joint_logits_new = np.concatenate([fv_x_scaled, fv_y_scaled], axis=1)

        Q_rate_x_scaled = -(Q_rate_x - np.max(Q_rate_x, axis=-1, keepdims=True))
        Q_rate_y_scaled = -(Q_rate_y - np.max(Q_rate_y, axis=-1, keepdims=True))
        joint_logits_new = (
            np.concatenate([Q_rate_x_scaled, Q_rate_y_scaled], axis=1) / self.ent_coef
        )

        # Apply Polyak damping if alpha < 1.0
        return (1 - alpha) * current_slice_logits + alpha * joint_logits_new


class ProjectedProbabilityDescent(DPPBasedIterations):
    """
    Direct Policy Gradient on the Probability Simplex.
    Steps raw probabilities along the gradient (-Q) and relies on
    the policy's built-in flooring and re-normalization.
    """

    def _update_policy_params(
        self, Q_rate_x, Q_rate_y, current_slice_params, n_x, pg_lr=0.1, **kwargs
    ):
        current_slice_probs = current_slice_params
        probs_x = current_slice_probs[:, :n_x]
        probs_y = current_slice_probs[:, n_x:]

        # Step in the negative direction of the action cost rate (-Q_rate)
        grad_x = -Q_rate_x
        grad_y = -Q_rate_y

        # If entropy coefficient is > 0, we can add the entropy gradient: -tau * (1 + log(pi))
        if self.ent_coef > 0:
            grad_x -= self.ent_coef * (1.0 + np.log(probs_x + 1e-12))
            grad_y -= self.ent_coef * (1.0 + np.log(probs_y + 1e-12))

        # Direct parameter update
        updated_probs_x = probs_x + pg_lr * grad_x
        updated_probs_y = probs_y + pg_lr * grad_y

        # Note: Flooring to 0 and sum-to-1 normalization are handled
        # afterward by self.policy.normalize_policy_()
        return np.concatenate([updated_probs_x, updated_probs_y], axis=1)


class ExactMirrorDescent(DPPBasedIterations):
    """
    Exact Mirror Descent.
    Updates logits by stepping in the direction of the First Variation.
    """

    def _update_policy_params(
        self, Q_rate_x, Q_rate_y, current_slice_params, n_x, md_lambda=0.5, **kwargs
    ):
        current_slice_logits = current_slice_params
        log_probs_x = current_slice_logits[:, :n_x]
        log_probs_y = current_slice_logits[:, n_x:]

        # First Variation = Q - tau * log(pi)
        uncentred_fv_x = (
            Q_rate_x
            - np.max(Q_rate_x, axis=-1, keepdims=True)
            + self.ent_coef * log_probs_x
        )
        uncentred_fv_y = (
            Q_rate_y
            - np.max(Q_rate_y, axis=-1, keepdims=True)
            + self.ent_coef * log_probs_y
        )

        # Add scaled first variation to current logits
        updated_logits_x = log_probs_x - md_lambda * uncentred_fv_x
        updated_logits_y = log_probs_y - md_lambda * uncentred_fv_y

        return np.concatenate([updated_logits_x, updated_logits_y], axis=1)


if __name__ == "__main__":
    print("Don't run directly, use run_dpp_based.py")
