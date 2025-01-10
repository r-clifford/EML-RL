import sys
import time
from gymnasium import Env
import onnxruntime as ort
import onnx
import torch as th
from stable_baselines3 import TD3, SAC
from eml_rl.utils import make_env

from sb3_contrib import TQC, CrossQ
from eml_rl.config.hyperparams.crossq_f1tenth import params as cq_params
from eml_rl.config.hyperparams.tqc_f1tenth import params as tqc_params

from eml_rl.config.hyperparams.td3_f1tenth import params as td3_params
from eml_rl.config.hyperparams.sac_f1tenth import params as sac_params


name = sys.argv[1]
path = sys.argv[2]
env = make_env("train", 0, 0)()
if name.lower() in "td3":
    model = TD3
    params = td3_params
elif name.lower() in "crossq":
    model = CrossQ
    params = cq_params
elif name.lower() in "tqc":
    model = TQC
    params = tqc_params
elif name.lower() in "sac":
    model = SAC
    params = sac_params
else:
    print("Invalid model type")
    exit(1)

model = model("MlpPolicy", env, **params)
if path:
    print(path)
    model = model.load(path, env)
model.policy.to("cpu")
# Note: by default model.policy.quantile_net.forward() returns quantiles
onnxable_model = model.policy
# observation_size = model.observation_space.shape[0]
observation_size = model.observation_space.shape

dummy_input = th.randn(1, *observation_size)
print(dummy_input.shape)
onnx_path = "model.onnx"
th.onnx.export(
    onnxable_model,
    dummy_input,
    onnx_path,
    opset_version=17,
    input_names=["input"],
)

# Load and test with onnx


onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

observation = dummy_input.cpu().numpy()
# providers = ort.get_available_providers()
providers = ["CPUExecutionProvider"]


def test(env: Env, ep):
    providers = [
        ep,
    ]
    print(ep)
    dummy_input = th.randn(1, *observation_size)
    observation = dummy_input.cpu().numpy()
    sess_options = ort.SessionOptions()
    # sess_options.enable_profiling = True
    ort_sess = ort.InferenceSession(
        "model.onnx", sess_options=sess_options, providers=providers
    )
    count = 10000
    count = 1
    start = time.time()
    observation = env.reset()
    action = ort_sess.run(None, {"input": observation})[0]
    while True:
        # seems to sometimes return speed < 0?
        action = ort_sess.run(None, {"input": observation})
        # action = model.predict(observation, deterministic=True)[0]
        # print(action)
        # print(base)
        observation, reward, _, _ = env.step(action)
        print(reward)
        env.render("human")
    end = time.time()
    print(f"{count}i")
    print(end - start)
    ips = count / (end - start)
    spi = 1 / ips
    print(f"{ips} /s")
    print(f"{spi} s/it")


for provider in providers:
    test(model.get_env(), provider)
