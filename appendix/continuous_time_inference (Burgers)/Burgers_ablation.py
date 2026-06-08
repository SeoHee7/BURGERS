"""
Ablation experiments for the continuous-time Burgers PINN example.

Put this file in:
    PINNs/appendix/continuous_time_inference (Burgers)/

Then run:
    python Burgers_ablation.py

Outputs:
    figures_ablation/ablation_results.csv
    figures_ablation/nu_ablation_metrics.png
    figures_ablation/nf_ablation_metrics.png
    figures_ablation/loss_ablation_metrics.png
    figures_ablation/loss_curves_*.png
    figures_ablation/loss_ablation_slices.png
"""

import os
import csv
import time
import json

import numpy as np
import scipy.io
from pyDOE import lhs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf


# -----------------------------
# Experiment configuration
# -----------------------------

SEED = 1234
NU_COEF = 0.01 / np.pi
LAYERS = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

BASE_N_U = 100
BASE_N_F = 10000

N_U_LIST = [20, 50, 100, 200, 400]
N_F_LIST = [1000, 5000, 10000, 20000]

LOSS_MODES = [
    ("data_only", "MSE_u only"),
    ("physics_only", "MSE_f only"),
    ("pinn", "MSE_u + MSE_f"),
]

OPT_OPTIONS = {
    "maxiter": 50000,
    "maxfun": 50000,
    "maxcor": 50,
    "maxls": 50,
    "ftol": 1.0 * np.finfo(float).eps,
}

OUT_DIR = "figures_ablation"


# Avoid requiring a LaTeX installation on AutoDL.
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
})


class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu, loss_mode="pinn"):
        self.lb = lb
        self.ub = ub
        self.loss_mode = loss_mode

        self.x_u = X_u[:, 0:1]
        self.t_u = X_u[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u = u
        self.layers = layers
        self.nu = nu
        self.loss_history = []

        self.weights, self.biases = self.initialize_NN(layers)

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_f = tf.reduce_mean(tf.square(self.f_pred))

        if loss_mode == "data_only":
            self.loss = self.loss_u
        elif loss_mode == "physics_only":
            self.loss = self.loss_f
        elif loss_mode == "pinn":
            self.loss = self.loss_u + self.loss_f
        else:
            raise ValueError("Unknown loss_mode: %s" % loss_mode)

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.loss,
            method="L-BFGS-B",
            options=OPT_OPTIONS,
        )

        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
        ))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(
            tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev),
            dtype=tf.float32,
        )

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u * u_x - self.nu * u_xx
        return f

    def callback(self, loss):
        self.loss_history.append(float(loss))
        if len(self.loss_history) % 50 == 0:
            print("Callback: %05d, Loss: %.3e" % (len(self.loss_history), loss))

    def train(self):
        tf_dict = {
            self.x_u_tf: self.x_u,
            self.t_u_tf: self.t_u,
            self.u_tf: self.u,
            self.x_f_tf: self.x_f,
            self.t_f_tf: self.t_f,
        }

        self.optimizer.minimize(
            self.sess,
            feed_dict=tf_dict,
            fetches=[self.loss],
            loss_callback=self.callback,
        )

    def predict(self, X_star):
        tf_dict = {
            self.x_u_tf: X_star[:, 0:1],
            self.t_u_tf: X_star[:, 1:2],
        }
        u_star = self.sess.run(self.u_pred, tf_dict)

        tf_dict = {
            self.x_f_tf: X_star[:, 0:1],
            self.t_f_tf: X_star[:, 1:2],
        }
        f_star = self.sess.run(self.f_pred, tf_dict)
        return u_star, f_star

    def evaluate_train_losses(self):
        tf_dict = {
            self.x_u_tf: self.x_u,
            self.t_u_tf: self.t_u,
            self.u_tf: self.u,
            self.x_f_tf: self.x_f,
            self.t_f_tf: self.t_f,
        }
        return self.sess.run([self.loss, self.loss_u, self.loss_f], tf_dict)

    def close(self):
        self.sess.close()


def load_burgers_data():
    data = scipy.io.loadmat("../Data/burgers_shock.mat")

    t = data["t"].flatten()[:, None]
    x = data["x"].flatten()[:, None]
    exact = np.real(data["usol"]).T

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = exact.flatten()[:, None]

    lb = X_star.min(0)
    ub = X_star.max(0)

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = exact[0:1, :].T

    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = exact[:, 0:1]

    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = exact[:, -1:]

    X_u_all = np.vstack([xx1, xx2, xx3])
    u_all = np.vstack([uu1, uu2, uu3])

    return {
        "x": x,
        "t": t,
        "X": X,
        "T": T,
        "exact": exact,
        "X_star": X_star,
        "u_star": u_star,
        "lb": lb,
        "ub": ub,
        "X_u_all": X_u_all,
        "u_all": u_all,
    }


def sample_training_data(data, n_u, n_f, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(data["X_u_all"].shape[0], n_u, replace=False)
    X_u_train = data["X_u_all"][idx, :]
    u_train = data["u_all"][idx, :]

    state = np.random.get_state()
    np.random.seed(seed + 100)
    X_f_train = data["lb"] + (data["ub"] - data["lb"]) * lhs(2, n_f)
    np.random.set_state(state)

    # Same as the original PINNs code: include initial/boundary points among
    # collocation points.
    X_f_train = np.vstack((X_f_train, X_u_train))
    return X_u_train, u_train, X_f_train


def reset_tf(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def run_one_experiment(data, ablation, name, n_u, n_f, loss_mode, seed):
    print("\n=== %s | %s | N_u=%d | N_f=%d | loss=%s ===" %
          (ablation, name, n_u, n_f, loss_mode))

    reset_tf(seed)
    X_u_train, u_train, X_f_train = sample_training_data(data, n_u, n_f, seed)

    model = PhysicsInformedNN(
        X_u_train,
        u_train,
        X_f_train,
        LAYERS,
        data["lb"],
        data["ub"],
        NU_COEF,
        loss_mode=loss_mode,
    )

    start_time = time.time()
    model.train()
    train_time = time.time() - start_time

    u_pred, _ = model.predict(data["X_star"])
    error_u = np.linalg.norm(data["u_star"] - u_pred, 2) / np.linalg.norm(data["u_star"], 2)
    final_loss, final_mse_u, final_mse_f = model.evaluate_train_losses()

    loss_history = np.asarray(model.loss_history, dtype=np.float64)
    if loss_history.size == 0:
        min_callback_loss = np.nan
        final_callback_loss = np.nan
    else:
        min_callback_loss = float(np.min(loss_history))
        final_callback_loss = float(loss_history[-1])

    result = {
        "ablation": ablation,
        "name": name,
        "N_u": n_u,
        "N_f": n_f,
        "N_f_total": X_f_train.shape[0],
        "loss_mode": loss_mode,
        "error_u": float(error_u),
        "final_loss": float(final_loss),
        "final_mse_u": float(final_mse_u),
        "final_mse_f": float(final_mse_f),
        "final_callback_loss": final_callback_loss,
        "min_callback_loss": min_callback_loss,
        "callback_steps": int(loss_history.size),
        "train_time_sec": float(train_time),
    }

    print(json.dumps(result, indent=2))
    model.close()

    U_pred = u_pred.reshape(data["X"].shape)
    return result, loss_history, U_pred


def ensure_out_dir():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)


def save_results_csv(results):
    path = os.path.join(OUT_DIR, "ablation_results.csv")
    fieldnames = [
        "ablation",
        "name",
        "N_u",
        "N_f",
        "N_f_total",
        "loss_mode",
        "error_u",
        "final_loss",
        "final_mse_u",
        "final_mse_f",
        "final_callback_loss",
        "min_callback_loss",
        "callback_steps",
        "train_time_sec",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Saved:", path)


def plot_metric_pair(rows, x_key, title, save_path):
    x = np.asarray([float(r[x_key]) for r in rows])
    error = np.asarray([float(r["error_u"]) for r in rows])
    final_loss = np.asarray([float(r["final_loss"]) for r in rows])
    train_time = np.asarray([float(r["train_time_sec"]) for r in rows])

    order = np.argsort(x)
    x = x[order]
    error = error[order]
    final_loss = final_loss[order]
    train_time = train_time[order]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=200)

    axes[0].plot(x, error, "o-", color="#1f77b4")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(x_key)
    axes[0].set_ylabel("Relative L2 error")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(x, final_loss, "o-", color="#d62728")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(x_key)
    axes[1].set_ylabel("Final training loss")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(x, train_time, "o-", color="#2ca02c")
    axes[2].set_xlabel(x_key)
    axes[2].set_ylabel("Training time (s)")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def plot_loss_ablation_metrics(rows, save_path):
    labels = [r["name"] for r in rows]
    x = np.arange(len(labels))
    width = 0.28

    error = np.asarray([float(r["error_u"]) for r in rows])
    mse_u = np.asarray([float(r["final_mse_u"]) for r in rows])
    mse_f = np.asarray([float(r["final_mse_f"]) for r in rows])
    train_time = np.asarray([float(r["train_time_sec"]) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=200)

    axes[0].bar(x, error, color="#1f77b4")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Relative L2 error")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(x - width / 2, mse_u, width, label="MSE_u", color="#ff7f0e")
    axes[1].bar(x + width / 2, mse_f, width, label="MSE_f", color="#2ca02c")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("Final loss component")
    axes[1].legend(frameon=False)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[2].bar(x, train_time, color="#9467bd")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    axes[2].set_ylabel("Training time (s)")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Loss ablation metrics")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def plot_loss_curves(histories, ablation, save_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    for label, hist in histories:
        if hist.size > 0:
            ax.plot(np.arange(1, hist.size + 1), hist, linewidth=1.4, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("L-BFGS callback step")
    ax.set_ylabel("Loss")
    ax.set_title("%s loss curves" % ablation)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def plot_loss_ablation_slices(data, predictions, save_path):
    times = [0.25, 0.50, 0.75]
    x = data["x"].flatten()
    t = data["t"].flatten()

    fig, axes = plt.subplots(1, len(times), figsize=(14, 3.8), dpi=200)
    for ax, time_value in zip(axes, times):
        idx = np.argmin(np.abs(t - time_value))
        ax.plot(x, data["exact"][idx, :], "k-", linewidth=2.0, label="Exact")
        for label, U_pred in predictions:
            ax.plot(x, U_pred[idx, :], "--", linewidth=1.5, label=label)
        ax.set_title("t = %.2f" % t[idx])
        ax.set_xlabel("x")
        ax.set_ylabel("u(t,x)")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Loss ablation: prediction slices")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


def main():
    ensure_out_dir()
    data = load_burgers_data()

    all_results = []
    histories_by_ablation = {
        "N_u": [],
        "N_f": [],
        "Loss": [],
    }
    loss_predictions = []

    # 1. N_u ablation.
    for n_u in N_U_LIST:
        name = "N_u=%d" % n_u
        result, hist, _ = run_one_experiment(
            data=data,
            ablation="N_u",
            name=name,
            n_u=n_u,
            n_f=BASE_N_F,
            loss_mode="pinn",
            seed=SEED + n_u,
        )
        all_results.append(result)
        histories_by_ablation["N_u"].append((name, hist))
        save_results_csv(all_results)

    # 2. N_f ablation.
    for n_f in N_F_LIST:
        name = "N_f=%d" % n_f
        result, hist, _ = run_one_experiment(
            data=data,
            ablation="N_f",
            name=name,
            n_u=BASE_N_U,
            n_f=n_f,
            loss_mode="pinn",
            seed=SEED + n_f,
        )
        all_results.append(result)
        histories_by_ablation["N_f"].append((name, hist))
        save_results_csv(all_results)

    # 3. Loss ablation.
    for offset, (loss_mode, label) in enumerate(LOSS_MODES):
        result, hist, U_pred = run_one_experiment(
            data=data,
            ablation="Loss",
            name=label,
            n_u=BASE_N_U,
            n_f=BASE_N_F,
            loss_mode=loss_mode,
            seed=SEED + 10000 + offset,
        )
        all_results.append(result)
        histories_by_ablation["Loss"].append((label, hist))
        loss_predictions.append((label, U_pred))
        save_results_csv(all_results)

    nu_rows = [r for r in all_results if r["ablation"] == "N_u"]
    nf_rows = [r for r in all_results if r["ablation"] == "N_f"]
    loss_rows = [r for r in all_results if r["ablation"] == "Loss"]

    plot_metric_pair(
        nu_rows,
        x_key="N_u",
        title="N_u ablation: supervised initial/boundary points",
        save_path=os.path.join(OUT_DIR, "nu_ablation_metrics.png"),
    )

    plot_metric_pair(
        nf_rows,
        x_key="N_f",
        title="N_f ablation: PDE collocation points",
        save_path=os.path.join(OUT_DIR, "nf_ablation_metrics.png"),
    )

    plot_loss_ablation_metrics(
        loss_rows,
        save_path=os.path.join(OUT_DIR, "loss_ablation_metrics.png"),
    )

    plot_loss_curves(
        histories_by_ablation["N_u"],
        "N_u ablation",
        os.path.join(OUT_DIR, "loss_curves_Nu.png"),
    )

    plot_loss_curves(
        histories_by_ablation["N_f"],
        "N_f ablation",
        os.path.join(OUT_DIR, "loss_curves_Nf.png"),
    )

    plot_loss_curves(
        histories_by_ablation["Loss"],
        "Loss ablation",
        os.path.join(OUT_DIR, "loss_curves_loss_ablation.png"),
    )

    plot_loss_ablation_slices(
        data,
        loss_predictions,
        os.path.join(OUT_DIR, "loss_ablation_slices.png"),
    )

    print("\nAll ablation experiments finished.")
    print("Results are saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
